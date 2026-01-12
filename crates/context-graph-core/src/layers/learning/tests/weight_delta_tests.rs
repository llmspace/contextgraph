//! Weight Delta tests - REAL implementations, NO MOCKS.

use crate::layers::learning::WeightDelta;

#[test]
fn test_weight_delta_magnitude() {
    let delta = WeightDelta {
        value: -0.5,
        surprise: 0.8,
        coherence_w: 0.6,
        learning_rate: 0.0005,
        was_clipped: false,
    };

    assert!((delta.magnitude() - 0.5).abs() < 1e-6);
    println!("[VERIFIED] WeightDelta.magnitude() = |value|");
}

#[test]
fn test_weight_delta_consolidation() {
    let delta = WeightDelta {
        value: 0.15,
        surprise: 0.8,
        coherence_w: 0.9,
        learning_rate: 0.5,
        was_clipped: false,
    };

    assert!(delta.should_consolidate(0.1));
    assert!(!delta.should_consolidate(0.2));
    println!("[VERIFIED] WeightDelta.should_consolidate() checks threshold");
}
