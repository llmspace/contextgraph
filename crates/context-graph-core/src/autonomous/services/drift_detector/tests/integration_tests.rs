//! Integration tests for DriftDetector - full scenarios and edge cases.

use crate::autonomous::drift::{DriftConfig, DriftSeverity, DriftTrend};
use crate::autonomous::services::drift_detector::{DetectorState, DriftDetector};

#[test]
fn test_negative_drift_detection() {
    // When alignment improves beyond baseline
    let mut detector = DriftDetector::new().with_baseline(0.70);
    detector.add_observation(0.85, 1000);

    // Drift score should still be positive (absolute value)
    assert!(
        (detector.get_drift_score() - 0.15).abs() < f32::EPSILON,
        "Drift score should be 0.15, got {}",
        detector.get_drift_score()
    );

    // Severity is based on absolute drift
    assert_eq!(detector.detect_drift(), DriftSeverity::Severe);
    println!("[PASS] test_negative_drift_detection");
}

#[test]
fn test_statistical_precision() {
    let mut detector = DriftDetector::new().with_baseline(0.50);

    // Add many observations to test numerical stability
    for i in 0..1000 {
        let alignment = 0.50 + 0.001 * (i % 10) as f32;
        detector.add_observation(alignment, i * 1000);
    }

    // Mean should be close to 0.5045 (average of 0.50 to 0.509)
    let expected_mean = 0.5045;
    assert!(
        (detector.rolling_mean() - expected_mean).abs() < 0.001,
        "Expected mean ~{}, got {}",
        expected_mean,
        detector.rolling_mean()
    );
    println!("[PASS] test_statistical_precision");
}

#[test]
fn test_state_serialization() {
    let mut detector = DriftDetector::new().with_baseline(0.80);
    detector.add_observation(0.75, 1000);
    detector.add_observation(0.70, 2000);

    let state = detector.state();
    let json = serde_json::to_string(state).expect("Failed to serialize state");
    let deserialized: DetectorState =
        serde_json::from_str(&json).expect("Failed to deserialize state");

    assert!((deserialized.rolling_mean - state.rolling_mean).abs() < f32::EPSILON);
    assert_eq!(deserialized.data_points.len(), state.data_points.len());
    println!("[PASS] test_state_serialization");
}

#[test]
fn test_boundary_alignment_values() {
    let mut detector = DriftDetector::new();

    // Test boundary values
    detector.add_observation(0.0, 1000);
    assert!((detector.state().current_alignment - 0.0).abs() < f32::EPSILON);

    detector.add_observation(1.0, 2000);
    assert!((detector.state().current_alignment - 1.0).abs() < f32::EPSILON);
    println!("[PASS] test_boundary_alignment_values");
}

#[test]
fn test_full_scenario_improving_from_severe() {
    let mut detector = DriftDetector::new().with_baseline(0.80).with_min_samples(3);

    // Start with severe drift
    detector.add_observation(0.60, 1000);
    detector.add_observation(0.60, 2000);
    detector.add_observation(0.60, 3000);
    assert_eq!(detector.detect_drift(), DriftSeverity::Severe);

    // Start improving
    for i in 4..15 {
        let alignment = 0.60 + ((i - 3) as f32 * 0.02);
        detector.add_observation(alignment.min(0.80), i * 1000);
    }

    // Trend should be improving
    assert_eq!(detector.compute_trend(), DriftTrend::Improving);
    println!("[PASS] test_full_scenario_improving_from_severe");
}

#[test]
fn test_all_severity_levels_accessible() {
    // Verify each severity level is reachable through actual observations
    let config = DriftConfig::default();

    // None
    let mut d = DriftDetector::with_config(config.clone()).with_baseline(0.75);
    d.add_observation(0.75, 1000);
    assert_eq!(d.detect_drift(), DriftSeverity::None);

    // Mild (> 0.01, < 0.05)
    let mut d = DriftDetector::with_config(config.clone()).with_baseline(0.75);
    d.add_observation(0.73, 1000); // 0.02 drift
    assert_eq!(d.detect_drift(), DriftSeverity::Mild);

    // Moderate (>= 0.05, < 0.10)
    let mut d = DriftDetector::with_config(config.clone()).with_baseline(0.75);
    d.add_observation(0.68, 1000); // 0.07 drift
    assert_eq!(d.detect_drift(), DriftSeverity::Moderate);

    // Severe (>= 0.10)
    let mut d = DriftDetector::with_config(config).with_baseline(0.75);
    d.add_observation(0.60, 1000); // 0.15 drift
    assert_eq!(d.detect_drift(), DriftSeverity::Severe);

    println!("[PASS] test_all_severity_levels_accessible");
}
