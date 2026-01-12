//! Trend detection tests for DriftDetector.

use crate::autonomous::drift::DriftTrend;
use crate::autonomous::services::drift_detector::DriftDetector;

#[test]
fn test_trend_stable() {
    let mut detector = DriftDetector::new().with_baseline(0.75);

    // Add stable values
    for i in 0..10 {
        detector.add_observation(0.75, i * 1000);
    }

    assert_eq!(detector.compute_trend(), DriftTrend::Stable);
    println!("[PASS] test_trend_stable");
}

#[test]
fn test_trend_improving() {
    let mut detector = DriftDetector::new().with_baseline(0.70);

    // Add increasing values
    for i in 0..10 {
        let alignment = 0.70 + (i as f32 * 0.02);
        detector.add_observation(alignment, i * 1000);
    }

    assert_eq!(detector.compute_trend(), DriftTrend::Improving);
    println!("[PASS] test_trend_improving");
}

#[test]
fn test_trend_declining() {
    let mut detector = DriftDetector::new().with_baseline(0.90);

    // Add decreasing values
    for i in 0..10 {
        let alignment = 0.90 - (i as f32 * 0.02);
        detector.add_observation(alignment, i * 1000);
    }

    assert_eq!(detector.compute_trend(), DriftTrend::Declining);
    println!("[PASS] test_trend_declining");
}

#[test]
fn test_trend_insufficient_data() {
    let mut detector = DriftDetector::new();
    detector.add_observation(0.80, 1000);
    detector.add_observation(0.75, 2000);

    // Less than 3 points should be stable
    assert_eq!(detector.compute_trend(), DriftTrend::Stable);
    println!("[PASS] test_trend_insufficient_data");
}

#[test]
fn test_linear_regression_slope_positive() {
    let detector = DriftDetector::new();

    // Clearly increasing: 0.1, 0.2, 0.3, 0.4, 0.5
    let values = vec![0.1, 0.2, 0.3, 0.4, 0.5];
    let slope = detector.compute_slope(&values);

    // Slope should be 0.1 per step
    assert!(
        (slope - 0.1).abs() < 1e-6,
        "Expected slope 0.1, got {}",
        slope
    );
    println!("[PASS] test_linear_regression_slope_positive");
}

#[test]
fn test_linear_regression_slope_negative() {
    let detector = DriftDetector::new();

    // Clearly decreasing: 0.5, 0.4, 0.3, 0.2, 0.1
    let values = vec![0.5, 0.4, 0.3, 0.2, 0.1];
    let slope = detector.compute_slope(&values);

    // Slope should be -0.1 per step
    assert!(
        (slope - (-0.1)).abs() < 1e-6,
        "Expected slope -0.1, got {}",
        slope
    );
    println!("[PASS] test_linear_regression_slope_negative");
}

#[test]
fn test_linear_regression_slope_zero() {
    let detector = DriftDetector::new();

    // Flat: all same values
    let values = vec![0.5, 0.5, 0.5, 0.5, 0.5];
    let slope = detector.compute_slope(&values);

    assert!(
        slope.abs() < f32::EPSILON,
        "Expected slope 0, got {}",
        slope
    );
    println!("[PASS] test_linear_regression_slope_zero");
}

#[test]
fn test_linear_regression_single_value() {
    let detector = DriftDetector::new();
    let values = vec![0.5];
    let slope = detector.compute_slope(&values);
    assert!((slope - 0.0).abs() < f32::EPSILON);
    println!("[PASS] test_linear_regression_single_value");
}
