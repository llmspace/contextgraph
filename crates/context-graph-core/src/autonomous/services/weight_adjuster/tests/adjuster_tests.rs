//! Unit tests for weight adjuster momentum, validation, and compute adjustment.

use crate::autonomous::bootstrap::GoalId;
use crate::autonomous::evolution::{
    AdjustmentReason as EvolutionAdjustmentReason, WeightAdjustment,
};
use crate::autonomous::services::weight_adjuster::{AdjustmentReason, WeightAdjuster};

// === Compute Momentum Tests ===

#[test]
fn test_compute_momentum_first_call() {
    let mut adjuster = WeightAdjuster::new();
    let goal_id = GoalId::new();

    // First call: v = 0.9 * 0 + 0.1 * 0.5 = 0.05
    let velocity = adjuster.compute_momentum(&goal_id, 0.5);
    assert!((velocity - 0.05).abs() < 1e-6);

    println!("[PASS] test_compute_momentum_first_call");
}

#[test]
fn test_compute_momentum_accumulation() {
    let mut adjuster = WeightAdjuster::new();
    let goal_id = GoalId::new();

    // First call with gradient 1.0
    // v1 = 0.9 * 0 + 0.1 * 1.0 = 0.1
    let v1 = adjuster.compute_momentum(&goal_id, 1.0);
    assert!((v1 - 0.1).abs() < 1e-6);

    // Second call with gradient 1.0
    // v2 = 0.9 * 0.1 + 0.1 * 1.0 = 0.09 + 0.1 = 0.19
    let v2 = adjuster.compute_momentum(&goal_id, 1.0);
    assert!((v2 - 0.19).abs() < 1e-6);

    // Third call with gradient 1.0
    // v3 = 0.9 * 0.19 + 0.1 * 1.0 = 0.171 + 0.1 = 0.271
    let v3 = adjuster.compute_momentum(&goal_id, 1.0);
    assert!((v3 - 0.271).abs() < 1e-6);

    println!("[PASS] test_compute_momentum_accumulation");
}

#[test]
fn test_compute_momentum_different_goals() {
    let mut adjuster = WeightAdjuster::new();
    let goal1 = GoalId::new();
    let goal2 = GoalId::new();

    let v1 = adjuster.compute_momentum(&goal1, 1.0);
    let v2 = adjuster.compute_momentum(&goal2, -1.0);

    assert!((v1 - 0.1).abs() < 1e-6);
    assert!((v2 - (-0.1)).abs() < 1e-6);

    // Check they're stored independently
    assert!(adjuster.get_velocity(&goal1).is_some());
    assert!(adjuster.get_velocity(&goal2).is_some());

    println!("[PASS] test_compute_momentum_different_goals");
}

#[test]
fn test_reset_momentum() {
    let mut adjuster = WeightAdjuster::new();
    let goal_id = GoalId::new();

    adjuster.compute_momentum(&goal_id, 1.0);
    assert!(adjuster.get_velocity(&goal_id).is_some());

    adjuster.reset_momentum(&goal_id);
    assert!(adjuster.get_velocity(&goal_id).is_none());

    println!("[PASS] test_reset_momentum");
}

#[test]
fn test_reset_all_momentum() {
    let mut adjuster = WeightAdjuster::new();
    let goal1 = GoalId::new();
    let goal2 = GoalId::new();

    adjuster.compute_momentum(&goal1, 1.0);
    adjuster.compute_momentum(&goal2, 1.0);

    adjuster.reset_all_momentum();

    assert!(adjuster.get_velocity(&goal1).is_none());
    assert!(adjuster.get_velocity(&goal2).is_none());

    println!("[PASS] test_reset_all_momentum");
}

// === Validate Adjustment Tests ===

#[test]
fn test_validate_adjustment_valid() {
    let adjuster = WeightAdjuster::new();

    let adjustment = WeightAdjustment {
        goal_id: GoalId::new(),
        old_weight: 0.5,
        new_weight: 0.6,
        reason: EvolutionAdjustmentReason::HighRetrievalActivity,
    };

    assert!(adjuster.validate_adjustment(&adjustment));

    println!("[PASS] test_validate_adjustment_valid");
}

#[test]
fn test_validate_adjustment_below_min() {
    let adjuster = WeightAdjuster::new();

    let adjustment = WeightAdjustment {
        goal_id: GoalId::new(),
        old_weight: 0.5,
        new_weight: 0.2, // Below min 0.3
        reason: EvolutionAdjustmentReason::LowActivity,
    };

    assert!(!adjuster.validate_adjustment(&adjustment));

    println!("[PASS] test_validate_adjustment_below_min");
}

#[test]
fn test_validate_adjustment_above_max() {
    let adjuster = WeightAdjuster::new();

    let adjustment = WeightAdjustment {
        goal_id: GoalId::new(),
        old_weight: 0.9,
        new_weight: 0.98, // Above max 0.95
        reason: EvolutionAdjustmentReason::HighRetrievalActivity,
    };

    assert!(!adjuster.validate_adjustment(&adjustment));

    println!("[PASS] test_validate_adjustment_above_max");
}

#[test]
fn test_validate_adjustment_negative_old_weight() {
    let adjuster = WeightAdjuster::new();

    let adjustment = WeightAdjustment {
        goal_id: GoalId::new(),
        old_weight: -0.1,
        new_weight: 0.5,
        reason: EvolutionAdjustmentReason::LowActivity,
    };

    assert!(!adjuster.validate_adjustment(&adjustment));

    println!("[PASS] test_validate_adjustment_negative_old_weight");
}

#[test]
fn test_validate_adjustment_nan() {
    let adjuster = WeightAdjuster::new();

    let adjustment = WeightAdjustment {
        goal_id: GoalId::new(),
        old_weight: 0.5,
        new_weight: f32::NAN,
        reason: EvolutionAdjustmentReason::HighRetrievalActivity,
    };

    assert!(!adjuster.validate_adjustment(&adjustment));

    let adjustment = WeightAdjustment {
        goal_id: GoalId::new(),
        old_weight: f32::NAN,
        new_weight: 0.5,
        reason: EvolutionAdjustmentReason::HighRetrievalActivity,
    };

    assert!(!adjuster.validate_adjustment(&adjustment));

    println!("[PASS] test_validate_adjustment_nan");
}

// === Compute Adjustment Tests ===

#[test]
fn test_compute_adjustment_high_performance() {
    let adjuster = WeightAdjuster::new();
    let goal_id = GoalId::new();

    // Performance 0.9 (0.4 above neutral 0.5) with lr=0.05
    // Expected delta: 0.4 * 0.05 = 0.02 increase
    let adjustment = adjuster.compute_adjustment(&goal_id, 0.9, 0.5);

    assert_eq!(adjustment.goal_id, goal_id);
    assert!((adjustment.old_weight - 0.5).abs() < f32::EPSILON);
    assert!((adjustment.new_weight - 0.52).abs() < 1e-6);

    println!("[PASS] test_compute_adjustment_high_performance");
}

#[test]
fn test_compute_adjustment_low_performance() {
    let adjuster = WeightAdjuster::new();
    let goal_id = GoalId::new();

    // Performance 0.1 (0.4 below neutral 0.5) with lr=0.05
    // Expected delta: 0.4 * 0.05 = 0.02 decrease
    let adjustment = adjuster.compute_adjustment(&goal_id, 0.1, 0.5);

    assert!((adjustment.old_weight - 0.5).abs() < f32::EPSILON);
    assert!((adjustment.new_weight - 0.48).abs() < 1e-6);

    println!("[PASS] test_compute_adjustment_low_performance");
}

#[test]
fn test_compute_adjustment_neutral_performance() {
    let adjuster = WeightAdjuster::new();
    let goal_id = GoalId::new();

    // Performance at neutral 0.5, no change
    let adjustment = adjuster.compute_adjustment(&goal_id, 0.5, 0.6);

    assert!((adjustment.new_weight - 0.6).abs() < f32::EPSILON);

    println!("[PASS] test_compute_adjustment_neutral_performance");
}

#[test]
fn test_compute_adjustment_clamps_to_bounds() {
    let adjuster = WeightAdjuster::new();
    let goal_id = GoalId::new();

    // Very low weight with low performance should clamp to min
    let adjustment = adjuster.compute_adjustment(&goal_id, 0.0, 0.31);
    assert!(adjustment.new_weight >= 0.3);

    // Very high weight with high performance should clamp to max
    let adjustment = adjuster.compute_adjustment(&goal_id, 1.0, 0.94);
    assert!(adjustment.new_weight <= 0.95);

    println!("[PASS] test_compute_adjustment_clamps_to_bounds");
}

// === AdjustmentReason Tests ===

#[test]
fn test_adjustment_reason_equality() {
    let r1 = AdjustmentReason::PerformanceBased {
        performance_delta: 0.5,
    };
    let r2 = AdjustmentReason::PerformanceBased {
        performance_delta: 0.5,
    };
    let r3 = AdjustmentReason::PerformanceBased {
        performance_delta: 0.3,
    };

    assert_eq!(r1, r2);
    assert_ne!(r1, r3);

    assert_eq!(
        AdjustmentReason::Scheduled { cycle_id: 1 },
        AdjustmentReason::Scheduled { cycle_id: 1 }
    );
    assert_ne!(
        AdjustmentReason::Scheduled { cycle_id: 1 },
        AdjustmentReason::Scheduled { cycle_id: 2 }
    );

    println!("[PASS] test_adjustment_reason_equality");
}

#[test]
fn test_adjustment_reason_description() {
    let r1 = AdjustmentReason::PerformanceBased {
        performance_delta: 0.15,
    };
    assert!(r1.description().contains("Performance-based"));
    assert!(r1.description().contains("0.15"));

    let r2 = AdjustmentReason::DriftCorrection {
        drift_magnitude: 0.25,
    };
    assert!(r2.description().contains("Drift correction"));
    assert!(r2.description().contains("0.25"));

    let r3 = AdjustmentReason::UserFeedback { magnitude: 0.8 };
    assert!(r3.description().contains("User feedback"));
    assert!(r3.description().contains("0.8"));

    let r4 = AdjustmentReason::EvolutionBased {
        evolution_score: 0.6,
    };
    assert!(r4.description().contains("Evolution-based"));
    assert!(r4.description().contains("0.6"));

    let r5 = AdjustmentReason::Scheduled { cycle_id: 42 };
    assert!(r5.description().contains("Scheduled"));
    assert!(r5.description().contains("42"));

    println!("[PASS] test_adjustment_reason_description");
}
