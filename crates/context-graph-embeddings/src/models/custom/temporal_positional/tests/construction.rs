//! Construction and configuration tests for TemporalPositionalModel.

use crate::error::EmbeddingError;
use crate::traits::EmbeddingModel;

use super::super::{TemporalPositionalModel, DEFAULT_BASE, TEMPORAL_POSITIONAL_DIMENSION};

// =========================================================================
// CONSTRUCTION TESTS
// =========================================================================

#[test]
fn test_new_creates_initialized_model() {
    let model = TemporalPositionalModel::new();

    println!("BEFORE: model created");
    println!("AFTER: is_initialized = {}", model.is_initialized());

    assert!(
        model.is_initialized(),
        "Custom model must be initialized immediately"
    );
    assert_eq!(model.base(), DEFAULT_BASE, "Must use default base");
}

#[test]
fn test_default_base_is_10000() {
    let model = TemporalPositionalModel::new();

    println!("Base: {}", model.base());

    assert_eq!(model.base(), DEFAULT_BASE, "Default base should be 10000.0");
    assert_eq!(model.base(), 10000.0);
}

#[test]
fn test_custom_base_valid() {
    let custom_base = 5000.0;
    let model = TemporalPositionalModel::with_base(custom_base).expect("Should succeed");

    assert_eq!(model.base(), custom_base);
}

#[test]
fn test_custom_base_minimum_valid() {
    let model = TemporalPositionalModel::with_base(2.0).expect("Should succeed");
    assert_eq!(model.base(), 2.0);
}

#[test]
fn test_custom_base_zero_invalid() {
    let result = TemporalPositionalModel::with_base(0.0);

    assert!(result.is_err(), "Zero base should fail");
    match result {
        Err(EmbeddingError::ConfigError { message }) => {
            assert!(
                message.contains("must be in range"),
                "Error should mention range"
            );
        }
        Err(other) => panic!("Expected ConfigError, got {:?}", other),
        Ok(_) => panic!("Expected ConfigError, got Ok"),
    }
}

#[test]
fn test_custom_base_one_invalid() {
    let result = TemporalPositionalModel::with_base(1.0);

    assert!(result.is_err(), "Base = 1.0 should fail (must be > 1.0)");
}

#[test]
fn test_custom_base_negative_invalid() {
    let result = TemporalPositionalModel::with_base(-100.0);

    assert!(result.is_err(), "Negative base should fail");
}

#[test]
fn test_custom_base_nan_invalid() {
    let result = TemporalPositionalModel::with_base(f32::NAN);

    assert!(result.is_err(), "NaN base should fail");
}

#[test]
fn test_custom_base_inf_invalid() {
    let result = TemporalPositionalModel::with_base(f32::INFINITY);

    assert!(result.is_err(), "Inf base should fail");
}

#[test]
fn test_default_impl() {
    let model = TemporalPositionalModel::default();

    assert!(model.is_initialized());
    assert_eq!(model.base(), DEFAULT_BASE);
}

// =========================================================================
// THREAD SAFETY TESTS
// =========================================================================

#[test]
fn test_model_is_send() {
    fn assert_send<T: Send>() {}
    assert_send::<TemporalPositionalModel>();
}

#[test]
fn test_model_is_sync() {
    fn assert_sync<T: Sync>() {}
    assert_sync::<TemporalPositionalModel>();
}

// =========================================================================
// CONSTANTS TESTS
// =========================================================================

#[test]
fn test_constants_are_correct() {
    assert_eq!(TEMPORAL_POSITIONAL_DIMENSION, 512);
    assert_eq!(DEFAULT_BASE, 10000.0);
}

#[test]
fn test_dimension_even_for_sin_cos_pairs() {
    // Must be even for sin/cos pairs
    assert_eq!(TEMPORAL_POSITIONAL_DIMENSION % 2, 0);
}
