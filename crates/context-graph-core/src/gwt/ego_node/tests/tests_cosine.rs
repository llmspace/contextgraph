//! Tests for cosine_similarity_13d function
//!
//! Tests mathematical properties and edge cases for cosine similarity.

#![allow(clippy::assertions_on_constants)] // Intentional assertions on constant thresholds for documentation

#[allow(deprecated)]
use super::super::IC_CRISIS_THRESHOLD;
use super::super::{
    cosine_similarity_13d, IC_CRITICAL_THRESHOLD, IC_HEALTHY_THRESHOLD, IC_WARNING_THRESHOLD,
};

#[test]
fn test_cosine_similarity_13d_identical_vectors() {
    let v1 = [1.0; 13];
    let v2 = [1.0; 13];

    let similarity = cosine_similarity_13d(&v1, &v2);

    // Identical vectors should have cosine = 1.0
    assert!(
        (similarity - 1.0).abs() < 1e-6,
        "Expected 1.0 for identical vectors, got {}",
        similarity
    );
}

#[test]
fn test_cosine_similarity_13d_opposite_vectors() {
    let v1 = [1.0; 13];
    let v2 = [-1.0; 13];

    let similarity = cosine_similarity_13d(&v1, &v2);

    // Opposite vectors should have cosine = -1.0
    assert!(
        (similarity - (-1.0)).abs() < 1e-6,
        "Expected -1.0 for opposite vectors, got {}",
        similarity
    );
}

#[test]
fn test_cosine_similarity_13d_orthogonal_vectors() {
    // Create orthogonal vectors
    let mut v1 = [0.0; 13];
    let mut v2 = [0.0; 13];
    v1[0] = 1.0; // Unit vector along first axis
    v2[1] = 1.0; // Unit vector along second axis

    let similarity = cosine_similarity_13d(&v1, &v2);

    // Orthogonal vectors should have cosine = 0.0
    assert!(
        similarity.abs() < 1e-6,
        "Expected 0.0 for orthogonal vectors, got {}",
        similarity
    );
}

#[test]
fn test_cosine_similarity_13d_zero_vector_first() {
    let v1 = [0.0; 13];
    let v2 = [1.0; 13];

    let similarity = cosine_similarity_13d(&v1, &v2);

    // Zero vector should return 0.0 per spec
    assert_eq!(
        similarity, 0.0,
        "Zero magnitude vector should return 0.0, got {}",
        similarity
    );
}

#[test]
fn test_cosine_similarity_13d_zero_vector_second() {
    let v1 = [1.0; 13];
    let v2 = [0.0; 13];

    let similarity = cosine_similarity_13d(&v1, &v2);

    // Zero vector should return 0.0 per spec
    assert_eq!(
        similarity, 0.0,
        "Zero magnitude vector should return 0.0, got {}",
        similarity
    );
}

#[test]
fn test_cosine_similarity_13d_both_zero_vectors() {
    let v1 = [0.0; 13];
    let v2 = [0.0; 13];

    let similarity = cosine_similarity_13d(&v1, &v2);

    // Both zero should return 0.0
    assert_eq!(
        similarity, 0.0,
        "Both zero vectors should return 0.0, got {}",
        similarity
    );
}

#[test]
fn test_cosine_similarity_13d_near_zero_vectors() {
    let v1 = [1e-10; 13];
    let v2 = [1.0; 13];

    let similarity = cosine_similarity_13d(&v1, &v2);

    // Near-zero should return 0.0 per spec
    assert_eq!(
        similarity, 0.0,
        "Near-zero magnitude should return 0.0, got {}",
        similarity
    );
}

#[test]
fn test_cosine_similarity_13d_different_magnitudes() {
    let v1 = [1.0; 13];
    let v2 = [10.0; 13]; // Same direction, different magnitude

    let similarity = cosine_similarity_13d(&v1, &v2);

    // Same direction should have cosine = 1.0 regardless of magnitude
    assert!(
        (similarity - 1.0).abs() < 1e-6,
        "Same direction vectors should have cosine = 1.0, got {}",
        similarity
    );
}

#[test]
fn test_cosine_similarity_13d_real_purpose_vectors() {
    // Realistic purpose vectors (values in [0, 1])
    let v1 = [
        0.85, 0.78, 0.92, 0.67, 0.73, 0.61, 0.88, 0.75, 0.81, 0.69, 0.84, 0.72, 0.79,
    ];
    let v2 = [
        0.82, 0.75, 0.89, 0.70, 0.76, 0.65, 0.85, 0.72, 0.78, 0.72, 0.81, 0.69, 0.82,
    ];

    let similarity = cosine_similarity_13d(&v1, &v2);

    // Similar purpose vectors should have high cosine
    assert!(
        similarity > 0.99,
        "Similar purpose vectors should have high cosine, got {}",
        similarity
    );
    assert!(
        similarity <= 1.0,
        "Cosine should be <= 1.0, got {}",
        similarity
    );
}

#[test]
fn test_cosine_similarity_13d_clamping() {
    // Test many random-ish vectors to ensure clamping works
    for i in 0..100 {
        let v1: [f32; 13] = std::array::from_fn(|j| ((i + j) as f32 * 0.1).sin());
        let v2: [f32; 13] = std::array::from_fn(|j| ((i + j + 5) as f32 * 0.15).cos());

        let similarity = cosine_similarity_13d(&v1, &v2);

        assert!(
            (-1.0..=1.0).contains(&similarity),
            "Cosine must be in [-1, 1], got {} at iteration {}",
            similarity,
            i
        );
    }
}

// =========================================================================
// IC Threshold Constants Tests (IDENTITY-002)
// =========================================================================

#[test]
fn test_ic_critical_threshold_value() {
    // Per IDENTITY-002: IC < 0.5 = CRITICAL
    assert_eq!(
        IC_CRITICAL_THRESHOLD, 0.5,
        "IC_CRITICAL_THRESHOLD should be 0.5 per IDENTITY-002"
    );
}

#[test]
fn test_ic_warning_threshold_value() {
    // Per IDENTITY-002: IC in [0.7, 0.9) = WARNING
    assert_eq!(
        IC_WARNING_THRESHOLD, 0.7,
        "IC_WARNING_THRESHOLD should be 0.7 per IDENTITY-002"
    );
}

#[test]
fn test_ic_healthy_threshold_value() {
    // Per IDENTITY-002: IC >= 0.9 = HEALTHY
    assert_eq!(
        IC_HEALTHY_THRESHOLD, 0.9,
        "IC_HEALTHY_THRESHOLD should be 0.9 per IDENTITY-002"
    );
}

#[test]
#[allow(deprecated)]
fn test_ic_crisis_threshold_deprecated_alias() {
    // IC_CRISIS_THRESHOLD is deprecated, but should equal IC_WARNING_THRESHOLD for backwards compat
    assert_eq!(
        IC_CRISIS_THRESHOLD, IC_WARNING_THRESHOLD,
        "Deprecated IC_CRISIS_THRESHOLD should equal IC_WARNING_THRESHOLD"
    );
}

#[test]
fn test_ic_thresholds_relationship() {
    // Thresholds should be properly ordered: CRITICAL < WARNING < HEALTHY
    assert!(
        IC_CRITICAL_THRESHOLD < IC_WARNING_THRESHOLD,
        "CRITICAL ({}) must be < WARNING ({})",
        IC_CRITICAL_THRESHOLD,
        IC_WARNING_THRESHOLD
    );
    assert!(
        IC_WARNING_THRESHOLD < IC_HEALTHY_THRESHOLD,
        "WARNING ({}) must be < HEALTHY ({})",
        IC_WARNING_THRESHOLD,
        IC_HEALTHY_THRESHOLD
    );
}

#[test]
fn test_ic_threshold_status_classification() {
    // Per IDENTITY-002 constitution rule:
    // IC < 0.5 = CRITICAL (must trigger dream consolidation)
    // IC in [0.5, 0.7) = DEGRADED
    // IC in [0.7, 0.9) = WARNING
    // IC >= 0.9 = HEALTHY

    // Test boundary: 0.49 is CRITICAL
    let ic_critical = 0.49_f32;
    assert!(
        ic_critical < IC_CRITICAL_THRESHOLD,
        "IC=0.49 should be CRITICAL (< 0.5)"
    );

    // Test boundary: 0.50 is DEGRADED (not CRITICAL)
    let ic_degraded = 0.50_f32;
    assert!(
        ic_degraded >= IC_CRITICAL_THRESHOLD && ic_degraded < IC_WARNING_THRESHOLD,
        "IC=0.50 should be DEGRADED ([0.5, 0.7))"
    );

    // Test boundary: 0.70 is WARNING
    let ic_warning = 0.70_f32;
    assert!(
        ic_warning >= IC_WARNING_THRESHOLD && ic_warning < IC_HEALTHY_THRESHOLD,
        "IC=0.70 should be WARNING ([0.7, 0.9))"
    );

    // Test boundary: 0.90 is HEALTHY
    let ic_healthy = 0.90_f32;
    assert!(
        ic_healthy >= IC_HEALTHY_THRESHOLD,
        "IC=0.90 should be HEALTHY (>= 0.9)"
    );
}
