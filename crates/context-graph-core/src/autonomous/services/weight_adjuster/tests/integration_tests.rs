//! Integration tests for weight adjuster apply, report, and workflow operations.

use crate::autonomous::bootstrap::GoalId;
use crate::autonomous::evolution::{
    AdjustmentReason as EvolutionAdjustmentReason, WeightAdjustment,
};
use crate::autonomous::services::weight_adjuster::{
    AdjustmentReport, WeightAdjuster, WeightAdjusterConfig,
};

// === Apply Adjustments Tests ===

#[test]
fn test_apply_adjustments_empty() {
    let mut adjuster = WeightAdjuster::new();

    let report = adjuster.apply_adjustments(&[]);

    assert_eq!(report.adjustments_applied, 0);
    assert_eq!(report.adjustments_skipped, 0);
    assert!((report.total_delta - 0.0).abs() < f32::EPSILON);
    assert!(!report.has_adjustments());
    assert!(report.all_successful());

    println!("[PASS] test_apply_adjustments_empty");
}

#[test]
fn test_apply_adjustments_all_valid() {
    let mut adjuster = WeightAdjuster::new();

    let adjustments = vec![
        WeightAdjustment {
            goal_id: GoalId::new(),
            old_weight: 0.5,
            new_weight: 0.6,
            reason: EvolutionAdjustmentReason::HighRetrievalActivity,
        },
        WeightAdjustment {
            goal_id: GoalId::new(),
            old_weight: 0.7,
            new_weight: 0.8,
            reason: EvolutionAdjustmentReason::HighRetrievalActivity,
        },
    ];

    let report = adjuster.apply_adjustments(&adjustments);

    assert_eq!(report.adjustments_applied, 2);
    assert_eq!(report.adjustments_skipped, 0);
    assert!((report.total_delta - 0.2).abs() < 1e-6); // 0.1 + 0.1
    assert!((report.avg_delta - 0.1).abs() < 1e-6);
    assert!((report.max_delta - 0.1).abs() < 1e-6);
    assert!(report.has_adjustments());
    assert!(report.all_successful());

    println!("[PASS] test_apply_adjustments_all_valid");
}

#[test]
fn test_apply_adjustments_some_invalid() {
    let mut adjuster = WeightAdjuster::new();

    let adjustments = vec![
        WeightAdjustment {
            goal_id: GoalId::new(),
            old_weight: 0.5,
            new_weight: 0.6,
            reason: EvolutionAdjustmentReason::HighRetrievalActivity,
        },
        WeightAdjustment {
            goal_id: GoalId::new(),
            old_weight: 0.5,
            new_weight: 0.1, // Below min 0.3 - invalid
            reason: EvolutionAdjustmentReason::LowActivity,
        },
        WeightAdjustment {
            goal_id: GoalId::new(),
            old_weight: 0.8,
            new_weight: 0.85,
            reason: EvolutionAdjustmentReason::HighRetrievalActivity,
        },
    ];

    let report = adjuster.apply_adjustments(&adjustments);

    assert_eq!(report.adjustments_applied, 2);
    assert_eq!(report.adjustments_skipped, 1);
    assert!((report.total_delta - 0.15).abs() < 1e-6); // 0.1 + 0.05
    assert!(report.has_adjustments());
    assert!(!report.all_successful());
    assert_eq!(report.skipped_goals.len(), 1);

    println!("[PASS] test_apply_adjustments_some_invalid");
}

#[test]
fn test_apply_adjustments_varying_deltas() {
    let mut adjuster = WeightAdjuster::new();

    let adjustments = vec![
        WeightAdjustment {
            goal_id: GoalId::new(),
            old_weight: 0.5,
            new_weight: 0.55, // delta = 0.05
            reason: EvolutionAdjustmentReason::HighRetrievalActivity,
        },
        WeightAdjustment {
            goal_id: GoalId::new(),
            old_weight: 0.6,
            new_weight: 0.8, // delta = 0.20
            reason: EvolutionAdjustmentReason::HighRetrievalActivity,
        },
        WeightAdjustment {
            goal_id: GoalId::new(),
            old_weight: 0.7,
            new_weight: 0.65, // delta = 0.05
            reason: EvolutionAdjustmentReason::LowActivity,
        },
    ];

    let report = adjuster.apply_adjustments(&adjustments);

    assert_eq!(report.adjustments_applied, 3);
    assert!((report.total_delta - 0.30).abs() < 1e-6);
    assert!((report.avg_delta - 0.10).abs() < 1e-6);
    assert!((report.max_delta - 0.20).abs() < 1e-6);

    println!("[PASS] test_apply_adjustments_varying_deltas");
}

// === AdjustmentReport Tests ===

#[test]
fn test_adjustment_report_new() {
    let report = AdjustmentReport::new();

    assert_eq!(report.adjustments_applied, 0);
    assert_eq!(report.adjustments_skipped, 0);
    assert!((report.total_delta - 0.0).abs() < f32::EPSILON);
    assert!((report.avg_delta - 0.0).abs() < f32::EPSILON);
    assert!((report.max_delta - 0.0).abs() < f32::EPSILON);
    assert!(report.adjusted_goals.is_empty());
    assert!(report.skipped_goals.is_empty());

    println!("[PASS] test_adjustment_report_new");
}

#[test]
fn test_adjustment_report_default() {
    let report = AdjustmentReport::default();

    assert!(!report.has_adjustments());
    assert!(report.all_successful());

    println!("[PASS] test_adjustment_report_default");
}

// === Integration Tests ===

#[test]
fn test_gradient_descent_convergence() {
    let config = WeightAdjusterConfig {
        learning_rate: 0.1,
        min_weight: 0.0,
        max_weight: 1.0,
        momentum: 0.0, // No momentum for simple convergence test
        adjustment_threshold: 0.001,
    };
    let adjuster = WeightAdjuster::with_config(config).unwrap();

    let target = 0.8;
    let mut current = 0.2;

    // Run gradient descent for 50 iterations
    for _ in 0..50 {
        current = adjuster.gradient_step(current, target, 0.1);
    }

    // Should converge close to target
    assert!((current - target).abs() < 0.01);

    println!("[PASS] test_gradient_descent_convergence");
}

#[test]
fn test_momentum_accumulates_velocity() {
    // With high momentum (0.9), velocity accumulates over iterations
    let config = WeightAdjusterConfig {
        momentum: 0.9,
        ..WeightAdjusterConfig::default()
    };
    let mut adjuster = WeightAdjuster::with_config(config).unwrap();

    let goal_id = GoalId::new();
    let gradient = 1.0;

    // First call: v1 = 0.9 * 0 + 0.1 * 1.0 = 0.1
    let v1 = adjuster.compute_momentum(&goal_id, gradient);
    assert!((v1 - 0.1).abs() < 1e-6);

    // Second call: v2 = 0.9 * 0.1 + 0.1 * 1.0 = 0.19
    let v2 = adjuster.compute_momentum(&goal_id, gradient);
    assert!((v2 - 0.19).abs() < 1e-6);
    assert!(v2 > v1); // Velocity increased

    // After many iterations, velocity should approach gradient / (1 - momentum) = 1.0
    // i.e., asymptote towards 1.0 for gradient=1.0, momentum=0.9
    let mut v_final = v2;
    for _ in 0..50 {
        v_final = adjuster.compute_momentum(&goal_id, gradient);
    }

    // Should approach 1.0 (exact limit is gradient when sum converges)
    // With momentum=0.9, the geometric series converges to ~1.0
    assert!(v_final > 0.9);
    assert!(v_final < 1.01);

    println!("[PASS] test_momentum_accumulates_velocity");
}

#[test]
fn test_full_adjustment_workflow() {
    let mut adjuster = WeightAdjuster::new();
    let goal1 = GoalId::new();
    let goal2 = GoalId::new();

    // Simulate performance feedback
    let adj1 = adjuster.compute_adjustment(&goal1, 0.8, 0.5); // High performance
    let adj2 = adjuster.compute_adjustment(&goal2, 0.3, 0.6); // Low performance

    // Validate performance-based direction
    assert!(adj1.new_weight > adj1.old_weight); // Should increase
    assert!(adj2.new_weight < adj2.old_weight); // Should decrease

    // Apply adjustments
    let report = adjuster.apply_adjustments(&[adj1.clone(), adj2.clone()]);

    assert_eq!(report.adjustments_applied, 2);
    assert!(report.has_adjustments());
    assert!(report.all_successful());

    println!("[PASS] test_full_adjustment_workflow");
}

#[test]
fn test_weight_bounds_respected_throughout() {
    let config = WeightAdjusterConfig {
        learning_rate: 0.5, // Large LR to test bounds
        min_weight: 0.3,
        max_weight: 0.7,
        momentum: 0.0,
        adjustment_threshold: 0.001,
    };
    let adjuster = WeightAdjuster::with_config(config).unwrap();

    // Extreme high performance on already high weight
    let adj = adjuster.compute_adjustment(&GoalId::new(), 1.0, 0.65);
    assert!(adj.new_weight <= 0.7);

    // Extreme low performance on already low weight
    let adj = adjuster.compute_adjustment(&GoalId::new(), 0.0, 0.35);
    assert!(adj.new_weight >= 0.3);

    println!("[PASS] test_weight_bounds_respected_throughout");
}
