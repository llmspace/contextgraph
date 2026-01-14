//! Tests for IdentityContinuityMonitor
//!
//! Tests IC monitoring, crisis detection, and serialization.

use super::super::{
    IdentityContinuityMonitor, IdentityStatus, IC_CRITICAL_THRESHOLD, IC_WARNING_THRESHOLD,
    MAX_PV_HISTORY_SIZE,
};

/// Helper: Create a purpose vector with uniform values
fn uniform_pv(val: f32) -> [f32; 13] {
    [val; 13]
}

#[test]
fn test_identity_continuity_monitor_new() {
    let monitor = IdentityContinuityMonitor::new_for_testing();

    assert!(monitor.is_empty());
    assert_eq!(monitor.history_len(), 0);
    assert_eq!(monitor.crisis_threshold(), IC_CRITICAL_THRESHOLD);
    assert!(monitor.last_result().is_none());
    assert!(monitor.identity_coherence().is_none());
    assert!(monitor.current_status().is_none());
    assert!(!monitor.is_in_crisis()); // No result = not in crisis
}

#[test]
fn test_identity_continuity_monitor_with_threshold() {
    let monitor = IdentityContinuityMonitor::with_threshold_for_testing(0.8);

    assert_eq!(monitor.crisis_threshold(), 0.8);
    assert!(monitor.is_empty());
}

#[test]
fn test_identity_continuity_monitor_with_threshold_clamping() {
    // Test high value clamping
    let monitor_high = IdentityContinuityMonitor::with_threshold_for_testing(1.5);
    assert_eq!(monitor_high.crisis_threshold(), 1.0);

    // Test negative value clamping
    let monitor_low = IdentityContinuityMonitor::with_threshold_for_testing(-0.5);
    assert_eq!(monitor_low.crisis_threshold(), 0.0);
}

#[test]
fn test_identity_continuity_monitor_with_capacity() {
    let monitor = IdentityContinuityMonitor::with_capacity_for_testing(50);

    assert_eq!(monitor.history().max_size, 50);
    assert!(monitor.is_empty());
}

#[test]
fn test_identity_continuity_monitor_default() {
    let monitor = IdentityContinuityMonitor::default();

    assert!(monitor.is_empty());
    assert_eq!(monitor.crisis_threshold(), IC_CRITICAL_THRESHOLD);
    assert_eq!(monitor.history().max_size, MAX_PV_HISTORY_SIZE);
}

#[test]
fn test_identity_continuity_monitor_first_vector() {
    let mut monitor = IdentityContinuityMonitor::new_for_testing();
    let pv = uniform_pv(0.8);

    // BEFORE
    assert!(monitor.is_empty());
    assert!(monitor.last_result().is_none());

    // EXECUTE
    let result = monitor.compute_continuity(&pv, 0.9, "First vector");

    // AFTER
    assert!(!monitor.is_empty());
    assert!(monitor.is_first_vector());
    assert_eq!(monitor.history_len(), 1);

    // First vector should return IC = 1.0, Healthy
    assert_eq!(result.identity_coherence, 1.0);
    assert_eq!(result.status, IdentityStatus::Healthy);

    // Getters should return values
    assert_eq!(monitor.identity_coherence(), Some(1.0));
    assert_eq!(monitor.current_status(), Some(IdentityStatus::Healthy));
    assert!(!monitor.is_in_crisis()); // 1.0 >= 0.7
}

#[test]
fn test_identity_continuity_monitor_second_vector_identical() {
    let mut monitor = IdentityContinuityMonitor::new_for_testing();
    let pv = uniform_pv(0.8);

    // First vector
    monitor.compute_continuity(&pv, 0.9, "First");

    // Second vector - same PV
    let result = monitor.compute_continuity(&pv, 0.95, "Second");

    // Identical vectors: cos = 1.0, IC = 1.0 * 0.95 = 0.95
    assert!(
        (result.recent_continuity - 1.0).abs() < 1e-6,
        "Identical PVs should have cos = 1.0"
    );
    assert!(
        (result.identity_coherence - 0.95).abs() < 1e-6,
        "IC should be 1.0 * 0.95 = 0.95"
    );
    assert_eq!(result.status, IdentityStatus::Healthy);
}

#[test]
fn test_identity_continuity_monitor_drift_detection() {
    let mut monitor = IdentityContinuityMonitor::new_for_testing();

    // First vector - high values
    let pv1 = uniform_pv(0.9);
    monitor.compute_continuity(&pv1, 0.95, "Initial");

    // Second vector - very different (drift)
    let pv2 = uniform_pv(0.1);
    let result = monitor.compute_continuity(&pv2, 0.95, "Drifted");

    // Different vectors: cos([0.9;13], [0.1;13]) = 1.0 (same direction, diff magnitude)
    // Wait, uniform vectors have same direction regardless of magnitude
    // Since both are uniform positive, they're parallel
    assert!((result.recent_continuity - 1.0).abs() < 1e-6);
}

#[test]
fn test_identity_continuity_monitor_real_drift() {
    let mut monitor = IdentityContinuityMonitor::new_for_testing();

    // First vector - realistic purpose vector
    let pv1 = [0.9, 0.85, 0.92, 0.8, 0.88, 0.75, 0.95, 0.82, 0.87, 0.78, 0.91, 0.83, 0.86];
    monitor.compute_continuity(&pv1, 0.95, "Aligned");

    // Second vector - shifted purpose (some dimensions change)
    let pv2 = [0.3, 0.25, 0.32, 0.9, 0.88, 0.95, 0.25, 0.92, 0.17, 0.98, 0.21, 0.93, 0.16];
    let result = monitor.compute_continuity(&pv2, 0.9, "Shifted");

    // These vectors have different patterns, so cos < 1.0
    assert!(
        result.recent_continuity < 1.0,
        "Different patterns should have cos < 1.0"
    );
}

#[test]
fn test_identity_continuity_monitor_low_kuramoto_r() {
    let mut monitor = IdentityContinuityMonitor::new_for_testing();
    let pv = uniform_pv(0.8);

    // First vector
    monitor.compute_continuity(&pv, 0.9, "First");

    // Second vector with low r (fragmented consciousness)
    let result = monitor.compute_continuity(&pv, 0.3, "Low sync");

    // cos = 1.0 (identical), r = 0.3, IC = 0.3
    assert!(
        (result.identity_coherence - 0.3).abs() < 1e-6,
        "IC should be 1.0 * 0.3 = 0.3"
    );
    assert_eq!(
        result.status,
        IdentityStatus::Critical,
        "IC=0.3 < 0.5 should be Critical"
    );
    assert!(monitor.is_in_crisis(), "IC=0.3 < 0.7 should be in crisis");
}

#[test]
fn test_identity_continuity_monitor_crisis_threshold_custom() {
    // More strict threshold
    let mut monitor = IdentityContinuityMonitor::with_threshold_for_testing(0.9);
    let pv = uniform_pv(0.8);

    monitor.compute_continuity(&pv, 0.95, "First");
    let result = monitor.compute_continuity(&pv, 0.85, "Second");

    // IC = 1.0 * 0.85 = 0.85
    assert!((result.identity_coherence - 0.85).abs() < 1e-6);

    // With threshold 0.9, IC=0.85 should be in crisis
    assert!(
        monitor.is_in_crisis(),
        "IC=0.85 < threshold=0.9 should be in crisis"
    );

    // Standard threshold would not be crisis
    assert!(
        result.identity_coherence >= IC_WARNING_THRESHOLD,
        "IC=0.85 >= standard threshold 0.7"
    );
}

#[test]
fn test_identity_continuity_monitor_history_accumulation() {
    let mut monitor = IdentityContinuityMonitor::with_capacity_for_testing(5);

    // Add 7 vectors
    for i in 0..7 {
        let pv = uniform_pv(0.5 + (i as f32 * 0.05));
        monitor.compute_continuity(&pv, 0.9, format!("Vector {}", i));
    }

    // Should only have 5 due to capacity
    assert_eq!(
        monitor.history_len(),
        5,
        "History should be capped at capacity 5"
    );
}

#[test]
fn test_identity_continuity_monitor_serialization() {
    let mut original = IdentityContinuityMonitor::with_threshold_for_testing(0.8);
    let pv1 = uniform_pv(0.75);
    let pv2 = uniform_pv(0.8);
    original.compute_continuity(&pv1, 0.9, "First");
    original.compute_continuity(&pv2, 0.85, "Second");

    // Serialize with bincode
    let serialized = bincode::serialize(&original).expect("Serialization should not fail");

    // Deserialize
    let restored: IdentityContinuityMonitor =
        bincode::deserialize(&serialized).expect("Deserialization should not fail");

    // Verify state preserved
    assert_eq!(restored.history_len(), original.history_len());
    assert_eq!(restored.crisis_threshold(), original.crisis_threshold());
    assert_eq!(restored.identity_coherence(), original.identity_coherence());
    assert_eq!(restored.current_status(), original.current_status());
}

#[test]
fn test_identity_continuity_monitor_json_serialization() {
    let mut original = IdentityContinuityMonitor::new_for_testing();
    original.compute_continuity(&uniform_pv(0.7), 0.9, "Test");

    // Serialize to JSON
    let json = serde_json::to_string(&original).expect("JSON serialization should not fail");

    // Deserialize from JSON
    let restored: IdentityContinuityMonitor =
        serde_json::from_str(&json).expect("JSON deserialization should not fail");

    assert_eq!(restored.history_len(), original.history_len());
}
