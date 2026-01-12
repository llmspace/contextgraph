//! Tests for KMeansConfig validation.

use crate::index::purpose::clustering::config::KMeansConfig;

#[test]
fn test_kmeans_config_valid() {
    let config = KMeansConfig::new(5, 100, 1e-6).unwrap();

    assert_eq!(config.k, 5);
    assert_eq!(config.max_iterations, 100);
    assert!((config.convergence_threshold - 1e-6).abs() < 1e-10);

    println!("[VERIFIED] KMeansConfig::new creates valid config");
}

#[test]
fn test_kmeans_config_with_k() {
    let config = KMeansConfig::with_k(10).unwrap();

    assert_eq!(config.k, 10);
    assert_eq!(config.max_iterations, 100);

    println!("[VERIFIED] KMeansConfig::with_k creates config with defaults");
}

#[test]
fn test_kmeans_config_invalid_k_zero() {
    let result = KMeansConfig::new(0, 100, 1e-6);

    assert!(result.is_err());
    let msg = result.unwrap_err().to_string();
    assert!(msg.contains("k must be > 0"));

    println!("[VERIFIED] FAIL FAST: KMeansConfig rejects k=0: {}", msg);
}

#[test]
fn test_kmeans_config_invalid_max_iterations_zero() {
    let result = KMeansConfig::new(5, 0, 1e-6);

    assert!(result.is_err());
    let msg = result.unwrap_err().to_string();
    assert!(msg.contains("max_iterations must be > 0"));

    println!(
        "[VERIFIED] FAIL FAST: KMeansConfig rejects max_iterations=0: {}",
        msg
    );
}

#[test]
fn test_kmeans_config_invalid_threshold_zero() {
    let result = KMeansConfig::new(5, 100, 0.0);

    assert!(result.is_err());
    let msg = result.unwrap_err().to_string();
    assert!(msg.contains("convergence_threshold must be > 0.0"));

    println!(
        "[VERIFIED] FAIL FAST: KMeansConfig rejects convergence_threshold=0.0: {}",
        msg
    );
}

#[test]
fn test_kmeans_config_invalid_threshold_negative() {
    let result = KMeansConfig::new(5, 100, -1e-6);

    assert!(result.is_err());

    println!("[VERIFIED] FAIL FAST: KMeansConfig rejects negative convergence_threshold");
}

#[test]
fn test_kmeans_config_invalid_threshold_nan() {
    let result = KMeansConfig::new(5, 100, f32::NAN);

    assert!(result.is_err());

    println!("[VERIFIED] FAIL FAST: KMeansConfig rejects NaN convergence_threshold");
}

#[test]
fn test_kmeans_config_invalid_threshold_infinity() {
    let result = KMeansConfig::new(5, 100, f32::INFINITY);

    assert!(result.is_err());

    println!("[VERIFIED] FAIL FAST: KMeansConfig rejects infinite convergence_threshold");
}

#[test]
fn test_kmeans_config_default() {
    let config = KMeansConfig::default();

    assert_eq!(config.k, 3);
    assert_eq!(config.max_iterations, 100);
    assert!((config.convergence_threshold - 1e-6).abs() < 1e-10);

    println!("[VERIFIED] KMeansConfig::default creates sensible defaults");
}
