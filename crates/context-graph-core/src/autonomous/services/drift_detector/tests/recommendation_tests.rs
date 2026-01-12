//! Recommendation generation tests for DriftDetector.

use crate::autonomous::drift::{DriftConfig, DriftSeverity, DriftTrend};
use crate::autonomous::services::drift_detector::{DriftDetector, DriftRecommendation};

#[test]
fn test_recommendation_no_action() {
    let mut detector = DriftDetector::new().with_baseline(0.75).with_min_samples(5);

    // Add enough stable observations
    for i in 0..10 {
        detector.add_observation(0.75, i * 1000);
    }

    assert_eq!(detector.get_recommendation(), DriftRecommendation::NoAction);
    println!("[PASS] test_recommendation_no_action");
}

#[test]
fn test_recommendation_monitor_insufficient_samples() {
    let mut detector = DriftDetector::new().with_min_samples(10);

    // Only 5 samples
    for i in 0..5 {
        detector.add_observation(0.75, i * 1000);
    }

    assert_eq!(detector.get_recommendation(), DriftRecommendation::Monitor);
    println!("[PASS] test_recommendation_monitor_insufficient_samples");
}

#[test]
fn test_recommendation_user_intervention() {
    let mut detector = DriftDetector::new().with_baseline(0.90).with_min_samples(5);

    // Add declining observations that create severe drift
    for i in 0..10 {
        let alignment = 0.70 - (i as f32 * 0.02);
        detector.add_observation(alignment.max(0.0), i * 1000);
    }

    assert_eq!(detector.detect_drift(), DriftSeverity::Severe);
    assert_eq!(detector.compute_trend(), DriftTrend::Declining);
    assert_eq!(
        detector.get_recommendation(),
        DriftRecommendation::UserIntervention
    );
    println!("[PASS] test_recommendation_user_intervention");
}

#[test]
fn test_recommendation_adjust_thresholds() {
    let mut detector = DriftDetector::new().with_baseline(0.80).with_min_samples(5);

    // Create moderate declining drift
    for i in 0..10 {
        let alignment = 0.75 - (i as f32 * 0.005);
        detector.add_observation(alignment, i * 1000);
    }

    assert_eq!(detector.detect_drift(), DriftSeverity::Moderate);
    assert_eq!(detector.compute_trend(), DriftTrend::Declining);
    assert_eq!(
        detector.get_recommendation(),
        DriftRecommendation::AdjustThresholds
    );
    println!("[PASS] test_recommendation_adjust_thresholds");
}

#[test]
fn test_reset_baseline() {
    let mut detector = DriftDetector::new().with_baseline(0.80);

    // Create drift
    for i in 0..5 {
        detector.add_observation(0.70, i * 1000);
    }

    assert_eq!(detector.detect_drift(), DriftSeverity::Severe);

    // Reset baseline to current mean
    detector.reset_baseline();

    assert!((detector.baseline() - 0.70).abs() < f32::EPSILON);
    assert_eq!(detector.detect_drift(), DriftSeverity::None);
    println!("[PASS] test_reset_baseline");
}

#[test]
fn test_window_trimming() {
    let config = DriftConfig {
        window_days: 1, // 1 day window
        ..Default::default()
    };
    let mut detector = DriftDetector::with_config(config);

    // Add observations spread across 2 days (in milliseconds)
    let day_ms = 24 * 60 * 60 * 1000u64;

    // Old observations (before window)
    for i in 0..5 {
        detector.add_observation(0.50, i * 1000);
    }

    // Jump forward past window
    let current_time = 2 * day_ms;
    for i in 0..5 {
        detector.add_observation(0.80, current_time + i * 1000);
    }

    // Only recent observations should remain
    assert_eq!(detector.data_point_count(), 5);

    // Mean should be based only on recent observations
    assert!(
        (detector.rolling_mean() - 0.80).abs() < f32::EPSILON,
        "Mean should be 0.80, got {}",
        detector.rolling_mean()
    );
    println!("[PASS] test_window_trimming");
}

#[test]
fn test_recommendation_serialization() {
    let recommendations = vec![
        DriftRecommendation::NoAction,
        DriftRecommendation::Monitor,
        DriftRecommendation::ReviewMemories,
        DriftRecommendation::AdjustThresholds,
        DriftRecommendation::RecalibrateBaseline,
        DriftRecommendation::UserIntervention,
    ];

    for rec in recommendations {
        let json = serde_json::to_string(&rec).expect("Failed to serialize");
        let deserialized: DriftRecommendation =
            serde_json::from_str(&json).expect("Failed to deserialize");
        assert_eq!(rec, deserialized);
    }
    println!("[PASS] test_recommendation_serialization");
}
