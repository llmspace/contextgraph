//! IdentityContinuity factory method and boundary tests
//!
//! Tests for IdentityContinuity::new(), first_vector(), is_in_crisis(), is_critical()

use chrono::Utc;

use super::super::{IdentityContinuity, IdentityStatus};

// =========================================================================
// IdentityContinuity Factory Method Tests
// =========================================================================

#[test]
fn test_identity_continuity_new_factory_computes_ic_correctly() {
    let result = IdentityContinuity::new(0.9, 0.85);

    let expected_ic = 0.9 * 0.85; // 0.765
    assert!(
        (result.identity_coherence - expected_ic).abs() < 1e-6,
        "IC should be {} but was {}",
        expected_ic,
        result.identity_coherence
    );
    assert_eq!(
        result.status,
        IdentityStatus::Warning,
        "IC=0.765 should be Warning (0.7 <= IC <= 0.9)"
    );
}

#[test]
fn test_identity_continuity_new_factory_clamps_negative_cosine() {
    let result = IdentityContinuity::new(-0.8, 0.9);

    assert!(
        result.identity_coherence >= 0.0,
        "IC must be >= 0, but was {}",
        result.identity_coherence
    );
    assert_eq!(
        result.status,
        IdentityStatus::Critical,
        "IC=0.0 should be Critical"
    );
}

#[test]
fn test_identity_continuity_new_factory_clamps_inputs() {
    let result = IdentityContinuity::new(1.5, 2.0);

    assert!(
        (result.recent_continuity - 1.0).abs() < 1e-6,
        "purpose_continuity should clamp to 1.0"
    );
    assert!(
        (result.kuramoto_order_parameter - 1.0).abs() < 1e-6,
        "kuramoto_r should clamp to 1.0"
    );
    assert!(
        (result.identity_coherence - 1.0).abs() < 1e-6,
        "IC should be 1.0 * 1.0 = 1.0"
    );
}

#[test]
fn test_identity_continuity_first_vector_returns_healthy() {
    let result = IdentityContinuity::first_vector();

    assert_eq!(result.identity_coherence, 1.0);
    assert_eq!(result.status, IdentityStatus::Healthy);
    assert_eq!(result.recent_continuity, 1.0);
    assert_eq!(result.kuramoto_order_parameter, 1.0);
}

#[test]
fn test_identity_continuity_first_vector_has_timestamp() {
    let before = Utc::now();
    let result = IdentityContinuity::first_vector();
    let after = Utc::now();

    assert!(
        result.computed_at >= before && result.computed_at <= after,
        "computed_at should be between test start and end"
    );
}

// =========================================================================
// is_in_crisis() and is_critical() Boundary Tests
// =========================================================================

#[test]
fn test_identity_continuity_is_in_crisis_boundary() {
    let at_boundary = IdentityContinuity::new(0.7, 1.0);
    assert!(
        !at_boundary.is_in_crisis(),
        "IC=0.7 should NOT be in crisis (boundary is < 0.7)"
    );

    let below_boundary = IdentityContinuity::new(0.699, 1.0);
    assert!(
        below_boundary.is_in_crisis(),
        "IC=0.699 should be in crisis"
    );
}

#[test]
fn test_identity_continuity_is_critical_boundary() {
    let at_boundary = IdentityContinuity::new(0.5, 1.0);
    assert!(
        !at_boundary.is_critical(),
        "IC=0.5 should NOT be critical (boundary is < 0.5)"
    );

    let below_boundary = IdentityContinuity::new(0.499, 1.0);
    assert!(below_boundary.is_critical(), "IC=0.499 should be critical");
}

#[test]
fn test_identity_continuity_crisis_methods_consistent_with_status() {
    let healthy = IdentityContinuity::new(1.0, 0.95);
    assert!(!healthy.is_in_crisis());
    assert!(!healthy.is_critical());
    assert_eq!(healthy.status, IdentityStatus::Healthy);

    let warning = IdentityContinuity::new(0.8, 1.0);
    assert!(!warning.is_in_crisis());
    assert!(!warning.is_critical());
    assert_eq!(warning.status, IdentityStatus::Warning);

    let degraded = IdentityContinuity::new(0.6, 1.0);
    assert!(degraded.is_in_crisis());
    assert!(!degraded.is_critical());
    assert_eq!(degraded.status, IdentityStatus::Degraded);

    let critical = IdentityContinuity::new(0.3, 1.0);
    assert!(critical.is_in_crisis());
    assert!(critical.is_critical());
    assert_eq!(critical.status, IdentityStatus::Critical);
}

// =========================================================================
// Serialization Tests
// =========================================================================

#[test]
fn test_identity_continuity_bincode_roundtrip() {
    let original = IdentityContinuity::new(0.85, 0.9);

    let serialized = bincode::serialize(&original).expect("Serialization must not fail");
    let deserialized: IdentityContinuity =
        bincode::deserialize(&serialized).expect("Deserialization must not fail");

    assert_eq!(original.identity_coherence, deserialized.identity_coherence);
    assert_eq!(original.recent_continuity, deserialized.recent_continuity);
    assert_eq!(
        original.kuramoto_order_parameter,
        deserialized.kuramoto_order_parameter
    );
    assert_eq!(original.status, deserialized.status);
    assert_eq!(original.computed_at, deserialized.computed_at);
}

#[test]
fn test_identity_continuity_json_roundtrip() {
    let original = IdentityContinuity::new(0.75, 0.8);

    let json = serde_json::to_string(&original).expect("JSON serialization must not fail");
    let deserialized: IdentityContinuity =
        serde_json::from_str(&json).expect("JSON deserialization must not fail");

    assert_eq!(original.identity_coherence, deserialized.identity_coherence);
    assert_eq!(original.status, deserialized.status);
}

// =========================================================================
// Edge Case Tests
// =========================================================================

#[test]
fn test_identity_continuity_zero_r_gives_critical() {
    let result = IdentityContinuity::new(1.0, 0.0);

    assert_eq!(result.identity_coherence, 0.0);
    assert_eq!(result.status, IdentityStatus::Critical);
    assert!(result.is_critical());
}

#[test]
fn test_identity_continuity_perfect_values() {
    let result = IdentityContinuity::new(1.0, 1.0);

    assert_eq!(result.identity_coherence, 1.0);
    assert_eq!(result.status, IdentityStatus::Healthy);
    assert!(!result.is_in_crisis());
    assert!(!result.is_critical());
}
