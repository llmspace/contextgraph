//! Core DriftDetector tests - basic operations, configuration, and observations.

use crate::autonomous::drift::{DriftConfig, DriftMonitoring, DriftSeverity};
use crate::autonomous::services::drift_detector::DriftDetector;

#[test]
fn test_drift_detector_new() {
    let detector = DriftDetector::new();
    assert_eq!(detector.data_point_count(), 0);
    assert!((detector.rolling_mean() - 0.75).abs() < f32::EPSILON);
    assert!((detector.baseline() - 0.75).abs() < f32::EPSILON);
    assert_eq!(detector.detect_drift(), DriftSeverity::None);
    println!("[PASS] test_drift_detector_new");
}

#[test]
fn test_drift_detector_with_config() {
    let config = DriftConfig {
        monitoring: DriftMonitoring::Periodic { interval_hours: 6 },
        alert_threshold: 0.03,
        auto_correct: false,
        severe_threshold: 0.08,
        window_days: 14,
    };
    let detector = DriftDetector::with_config(config);
    assert!(!detector.is_continuous_monitoring());
    assert!((detector.config().alert_threshold - 0.03).abs() < f32::EPSILON);
    assert!((detector.config().severe_threshold - 0.08).abs() < f32::EPSILON);
    println!("[PASS] test_drift_detector_with_config");
}

#[test]
fn test_drift_detector_with_baseline() {
    let detector = DriftDetector::new().with_baseline(0.85);
    assert!((detector.baseline() - 0.85).abs() < f32::EPSILON);
    assert!((detector.rolling_mean() - 0.85).abs() < f32::EPSILON);
    println!("[PASS] test_drift_detector_with_baseline");
}

#[test]
#[should_panic(expected = "FAIL FAST: baseline must be in range")]
fn test_drift_detector_invalid_baseline() {
    DriftDetector::new().with_baseline(1.5);
}

#[test]
#[should_panic(expected = "FAIL FAST: min_samples must be > 0")]
fn test_drift_detector_zero_min_samples() {
    DriftDetector::new().with_min_samples(0);
}

#[test]
fn test_add_observation_basic() {
    let mut detector = DriftDetector::new().with_baseline(0.80);

    detector.add_observation(0.75, 1000);
    assert_eq!(detector.data_point_count(), 1);
    assert!((detector.state().current_alignment - 0.75).abs() < f32::EPSILON);
    assert!((detector.rolling_mean() - 0.75).abs() < f32::EPSILON);
    println!("[PASS] test_add_observation_basic");
}

#[test]
#[should_panic(expected = "FAIL FAST: alignment must be in range")]
fn test_add_observation_invalid_alignment() {
    let mut detector = DriftDetector::new();
    detector.add_observation(-0.1, 1000);
}

#[test]
fn test_rolling_mean_calculation() {
    let mut detector = DriftDetector::new().with_baseline(0.80);

    detector.add_observation(0.90, 1000);
    detector.add_observation(0.80, 2000);
    detector.add_observation(0.70, 3000);

    // Mean should be (0.90 + 0.80 + 0.70) / 3 = 0.80
    let expected_mean = 0.80;
    assert!(
        (detector.rolling_mean() - expected_mean).abs() < 1e-6,
        "Expected mean {}, got {}",
        expected_mean,
        detector.rolling_mean()
    );
    println!("[PASS] test_rolling_mean_calculation");
}

#[test]
fn test_rolling_variance_calculation() {
    let mut detector = DriftDetector::new().with_baseline(0.80);

    // Add values with known variance: [0.70, 0.80, 0.90]
    // Mean = 0.80
    // Variance = ((0.70-0.80)^2 + (0.80-0.80)^2 + (0.90-0.80)^2) / 2
    //          = (0.01 + 0 + 0.01) / 2 = 0.01
    detector.add_observation(0.70, 1000);
    detector.add_observation(0.80, 2000);
    detector.add_observation(0.90, 3000);

    let expected_variance = 0.01;
    assert!(
        (detector.rolling_variance() - expected_variance).abs() < 1e-6,
        "Expected variance {}, got {}",
        expected_variance,
        detector.rolling_variance()
    );
    println!("[PASS] test_rolling_variance_calculation");
}

#[test]
fn test_variance_with_identical_values() {
    let mut detector = DriftDetector::new();

    // All same values should have zero variance
    for i in 0..5 {
        detector.add_observation(0.75, i * 1000);
    }

    assert!(
        detector.rolling_variance().abs() < f32::EPSILON,
        "Variance should be 0 for identical values, got {}",
        detector.rolling_variance()
    );
    println!("[PASS] test_variance_with_identical_values");
}

#[test]
fn test_drift_severity_none() {
    let mut detector = DriftDetector::new().with_baseline(0.75);
    detector.add_observation(0.75, 1000);

    assert_eq!(detector.detect_drift(), DriftSeverity::None);
    assert!(!detector.requires_attention());
    assert!(!detector.requires_intervention());
    println!("[PASS] test_drift_severity_none");
}

#[test]
fn test_drift_severity_mild() {
    let mut detector = DriftDetector::new().with_baseline(0.80);
    // Drift of 0.02 is mild (< 0.05 alert threshold)
    detector.add_observation(0.78, 1000);

    assert_eq!(detector.detect_drift(), DriftSeverity::Mild);
    assert!(!detector.requires_attention());
    assert!(!detector.requires_intervention());
    println!("[PASS] test_drift_severity_mild");
}

#[test]
fn test_drift_severity_moderate() {
    let mut detector = DriftDetector::new().with_baseline(0.80);
    // Drift of 0.07 is moderate (>= 0.05, < 0.10)
    detector.add_observation(0.73, 1000);

    assert_eq!(detector.detect_drift(), DriftSeverity::Moderate);
    assert!(detector.requires_attention());
    assert!(!detector.requires_intervention());
    println!("[PASS] test_drift_severity_moderate");
}

#[test]
fn test_drift_severity_severe() {
    let mut detector = DriftDetector::new().with_baseline(0.80);
    // Drift of 0.15 is severe (>= 0.10)
    detector.add_observation(0.65, 1000);

    assert_eq!(detector.detect_drift(), DriftSeverity::Severe);
    assert!(detector.requires_attention());
    assert!(detector.requires_intervention());
    println!("[PASS] test_drift_severity_severe");
}

#[test]
fn test_drift_score_calculation() {
    let mut detector = DriftDetector::new().with_baseline(0.80);
    detector.add_observation(0.70, 1000);

    let drift_score = detector.get_drift_score();
    assert!(
        (drift_score - 0.10).abs() < f32::EPSILON,
        "Expected drift score 0.10, got {}",
        drift_score
    );
    println!("[PASS] test_drift_score_calculation");
}

#[test]
fn test_continuous_monitoring_flag() {
    let detector_continuous = DriftDetector::new();
    assert!(detector_continuous.is_continuous_monitoring());

    let config = DriftConfig {
        monitoring: DriftMonitoring::Manual,
        ..Default::default()
    };
    let detector_manual = DriftDetector::with_config(config);
    assert!(!detector_manual.is_continuous_monitoring());
    println!("[PASS] test_continuous_monitoring_flag");
}

#[test]
fn test_custom_thresholds() {
    let config = DriftConfig {
        alert_threshold: 0.02,
        severe_threshold: 0.05,
        ..Default::default()
    };
    let mut detector = DriftDetector::with_config(config).with_baseline(0.80);

    // 0.03 drift should be moderate with custom thresholds
    detector.add_observation(0.77, 1000);
    assert_eq!(detector.detect_drift(), DriftSeverity::Moderate);

    // 0.06 drift should be severe with custom thresholds
    let mut detector2 = DriftDetector::with_config(DriftConfig {
        alert_threshold: 0.02,
        severe_threshold: 0.05,
        ..Default::default()
    })
    .with_baseline(0.80);
    detector2.add_observation(0.74, 1000);
    assert_eq!(detector2.detect_drift(), DriftSeverity::Severe);
    println!("[PASS] test_custom_thresholds");
}

#[test]
fn test_data_point_delta_from_mean() {
    let mut detector = DriftDetector::new().with_baseline(0.75);

    detector.add_observation(0.80, 1000);

    let point = detector.state().data_points.front().unwrap();
    // First observation: delta = 0.80 - 0.75 (initial mean) = 0.05
    assert!(
        (point.delta_from_mean - 0.05).abs() < f32::EPSILON,
        "Expected delta 0.05, got {}",
        point.delta_from_mean
    );
    println!("[PASS] test_data_point_delta_from_mean");
}
