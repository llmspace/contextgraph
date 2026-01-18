//! Synchronous unit tests for alignment types and computations.
//!
//! Tests individual components: weights, thresholds, scores, patterns, config, errors.

use super::fixtures::*;
use crate::alignment::*;
use crate::purpose::GoalLevel;
use crate::types::fingerprint::{AlignmentThreshold, PurposeVector, NUM_EMBEDDERS};

use uuid::Uuid;

// =============================================================================
// UNIT TESTS
// =============================================================================

#[test]
fn test_level_weights_invariant() {
    println!("\n============================================================");
    println!("TEST: test_level_weights_invariant");
    println!("============================================================");

    let weights = LevelWeights::default();

    println!("\nBEFORE STATE:");
    println!("  - strategic: {}", weights.strategic);
    println!("  - tactical: {}", weights.tactical);
    println!("  - immediate: {}", weights.immediate);

    let sum = weights.sum();
    println!("\nAFTER STATE:");
    println!("  - sum: {}", sum);
    println!("  - validate(): {:?}", weights.validate());

    assert!(
        (sum - 1.0).abs() < 0.001,
        "FAIL: Weights must sum to 1.0, got {}",
        sum
    );
    assert!(weights.validate().is_ok(), "FAIL: Validation failed");

    println!("\n[VERIFIED] LevelWeights invariant (sum=1.0) holds");
}

#[test]
fn test_alignment_threshold_classification() {
    println!("\n============================================================");
    println!("TEST: test_alignment_threshold_classification");
    println!("============================================================");

    let test_cases = [
        (0.80, AlignmentThreshold::Optimal, "Optimal"),
        (0.75, AlignmentThreshold::Optimal, "Optimal boundary"),
        (0.74, AlignmentThreshold::Acceptable, "Acceptable"),
        (0.70, AlignmentThreshold::Acceptable, "Acceptable boundary"),
        (0.69, AlignmentThreshold::Warning, "Warning"),
        (0.55, AlignmentThreshold::Warning, "Warning boundary"),
        (0.54, AlignmentThreshold::Critical, "Critical"),
        (0.30, AlignmentThreshold::Critical, "Deep Critical"),
    ];

    println!("\nTHRESHOLD CLASSIFICATION:");
    for (value, expected, desc) in test_cases {
        let actual = AlignmentThreshold::classify(value);
        println!(
            "  - {:.2} -> {:?} (expected {:?}) [{}]",
            value, actual, expected, desc
        );
        assert_eq!(
            actual, expected,
            "FAIL: {:.2} should be {:?}, got {:?}",
            value, expected, actual
        );
    }

    println!("\n[VERIFIED] AlignmentThreshold classification correct");
}

#[test]
fn test_goal_score_weighted_contribution() {
    println!("\n============================================================");
    println!("TEST: test_goal_score_weighted_contribution");
    println!("============================================================");

    // Use literal weights from LevelWeights::default() for clarity
    // Strategic=0.4, Tactical=0.3, Immediate=0.2 (3-level hierarchy per PRD v6)
    let test_cases = [
        (GoalLevel::Strategic, 0.8, 0.4, 0.32),
        (GoalLevel::Strategic, 0.7, 0.3, 0.21),
        (GoalLevel::Tactical, 0.6, 0.2, 0.12),
        (GoalLevel::Immediate, 0.5, 0.1, 0.05),
    ];

    println!("\nWEIGHTED CONTRIBUTIONS:");
    for (level, alignment, weight, expected_contrib) in test_cases {
        let goal_id = Uuid::new_v4();
        let score = GoalScore::new(goal_id, level, alignment, weight);

        println!(
            "  - {:?}: alignment={:.2} * weight={:.2} = {:.3} (expected {:.3})",
            level, alignment, weight, score.weighted_contribution, expected_contrib
        );

        assert!(
            (score.weighted_contribution - expected_contrib).abs() < 0.001,
            "FAIL: Weighted contribution mismatch for {:?}",
            level
        );
    }

    println!("\n[VERIFIED] GoalScore weighted contribution calculation correct");
}

#[test]
fn test_misalignment_flags_severity_levels() {
    println!("\n============================================================");
    println!("TEST: test_misalignment_flags_severity_levels");
    println!("============================================================");

    // No flags = severity 0
    let flags_none = MisalignmentFlags::empty();
    assert_eq!(flags_none.severity(), 0, "FAIL: Empty should be 0");
    println!("  - empty flags: severity = {}", flags_none.severity());

    // Warning flags = severity 1
    let mut flags_warn = MisalignmentFlags::empty();
    flags_warn.tactical_without_strategic = true;
    assert_eq!(flags_warn.severity(), 1, "FAIL: Warning should be 1");
    println!(
        "  - tactical_without_strategic: severity = {}",
        flags_warn.severity()
    );

    // Critical flags = severity 2
    let mut flags_crit = MisalignmentFlags::empty();
    flags_crit.mark_below_threshold(Uuid::new_v4());
    assert_eq!(flags_crit.severity(), 2, "FAIL: Critical should be 2");
    println!("  - below_threshold: severity = {}", flags_crit.severity());

    // Divergent = severity 2
    let mut flags_div = MisalignmentFlags::empty();
    flags_div.mark_divergent(Uuid::new_v4(), Uuid::new_v4());
    assert_eq!(flags_div.severity(), 2, "FAIL: Divergent should be 2");
    println!(
        "  - divergent_hierarchy: severity = {}",
        flags_div.severity()
    );

    println!("\n[VERIFIED] MisalignmentFlags severity levels correct");
}

#[test]
fn test_pattern_type_classification() {
    println!("\n============================================================");
    println!("TEST: test_pattern_type_classification");
    println!("============================================================");

    let positive_patterns = [
        PatternType::OptimalAlignment,
        PatternType::HierarchicalCoherence,
    ];

    let negative_patterns = [
        PatternType::TacticalWithoutStrategic,
        PatternType::DivergentHierarchy,
        PatternType::CriticalMisalignment,
        PatternType::InconsistentAlignment,
        PatternType::StrategicDrift,
    ];

    println!("\nPOSITIVE PATTERNS:");
    for p in &positive_patterns {
        assert!(p.is_positive(), "FAIL: {:?} should be positive", p);
        assert!(!p.is_negative(), "FAIL: {:?} should not be negative", p);
        println!(
            "  - {:?}: is_positive=true, severity={}",
            p,
            p.default_severity()
        );
    }

    println!("\nNEGATIVE PATTERNS:");
    for p in &negative_patterns {
        assert!(p.is_negative(), "FAIL: {:?} should be negative", p);
        assert!(!p.is_positive(), "FAIL: {:?} should not be positive", p);
        println!(
            "  - {:?}: is_negative=true, severity={}",
            p,
            p.default_severity()
        );
    }

    println!("\n[VERIFIED] PatternType classification correct");
}

#[test]
fn test_embedder_breakdown_statistics() {
    println!("\n============================================================");
    println!("TEST: test_embedder_breakdown_statistics");
    println!("============================================================");

    // Create purpose vector with varying alignments
    let mut alignments = [0.7; NUM_EMBEDDERS];
    alignments[0] = 0.95; // Best
    alignments[5] = 0.40; // Worst (critical)
    alignments[8] = 0.60; // Warning

    let pv = PurposeVector::new(alignments);
    let breakdown = EmbedderBreakdown::from_purpose_vector(&pv);

    println!("\nBEFORE STATE:");
    println!("  - alignments: {:?}", alignments);

    println!("\nAFTER STATE:");
    println!(
        "  - best_embedder: {} ({})",
        breakdown.best_embedder,
        EmbedderBreakdown::embedder_name(breakdown.best_embedder)
    );
    println!(
        "  - worst_embedder: {} ({})",
        breakdown.worst_embedder,
        EmbedderBreakdown::embedder_name(breakdown.worst_embedder)
    );
    println!("  - mean: {:.3}", breakdown.mean);
    println!("  - std_dev: {:.3}", breakdown.std_dev);

    let (optimal, acceptable, warning, critical) = breakdown.threshold_counts();
    println!("  - optimal count: {}", optimal);
    println!("  - acceptable count: {}", acceptable);
    println!("  - warning count: {}", warning);
    println!("  - critical count: {}", critical);

    assert_eq!(breakdown.best_embedder, 0, "FAIL: Best should be index 0");
    assert_eq!(breakdown.worst_embedder, 5, "FAIL: Worst should be index 5");
    assert!(breakdown.std_dev > 0.0, "FAIL: std_dev should be positive");

    let misaligned = breakdown.misaligned_embedders();
    println!("  - misaligned embedders: {:?}", misaligned);
    assert!(
        misaligned.len() >= 2,
        "FAIL: Should have at least 2 misaligned"
    );

    println!("\n[VERIFIED] EmbedderBreakdown statistics correct");
}

#[test]
fn test_goal_alignment_score_composite_computation() {
    println!("\n============================================================");
    println!("TEST: test_goal_alignment_score_composite_computation");
    println!("============================================================");

    // TASK-P0-001: Updated for 3-level hierarchy
    // GoalScore weights are used for individual score contribution tracking,
    // but composite uses LevelWeights::default() = {strategic: 0.5, tactical: 0.3, immediate: 0.2}
    let scores = vec![
        GoalScore::new(Uuid::new_v4(), GoalLevel::Strategic, 0.90, 0.4),
        GoalScore::new(Uuid::new_v4(), GoalLevel::Strategic, 0.80, 0.3),
        GoalScore::new(Uuid::new_v4(), GoalLevel::Tactical, 0.70, 0.2),
        GoalScore::new(Uuid::new_v4(), GoalLevel::Immediate, 0.60, 0.1),
    ];

    let weights = LevelWeights::default();

    println!("\nBEFORE STATE:");
    for s in &scores {
        println!(
            "  - {:?} {}: alignment={:.2}",
            s.level, s.goal_id, s.alignment
        );
    }

    let result = GoalAlignmentScore::compute(scores, weights);

    // TASK-P0-001: Composite uses LevelWeights::default() = {strategic: 0.5, tactical: 0.3, immediate: 0.2}
    // strategic_alignment = avg(0.90, 0.80) = 0.85
    // composite = (0.5 * 0.85 + 0.3 * 0.70 + 0.2 * 0.60) / 1.0 = 0.755
    let expected = 0.755;

    println!("\nAFTER STATE:");
    println!(
        "  - composite_score: {:.3} (expected {:.3})",
        result.composite_score, expected
    );
    println!("  - strategic_alignment: {:.3}", result.strategic_alignment);
    println!("  - tactical_alignment: {:.3}", result.tactical_alignment);
    println!("  - immediate_alignment: {:.3}", result.immediate_alignment);
    println!("  - threshold: {:?}", result.threshold);

    assert!(
        (result.composite_score - expected).abs() < 0.01,
        "FAIL: Composite score mismatch"
    );
    // strategic_alignment is the AVERAGE of all Strategic scores: (0.90 + 0.80) / 2 = 0.85
    assert!(
        (result.strategic_alignment - 0.85).abs() < 0.001,
        "FAIL: Strategic alignment should be average of 0.90 and 0.80 = 0.85"
    );
    assert!(
        (result.tactical_alignment - 0.70).abs() < 0.001,
        "FAIL: Tactical alignment mismatch"
    );
    assert!(
        (result.immediate_alignment - 0.60).abs() < 0.001,
        "FAIL: Immediate alignment mismatch"
    );

    println!("\n[VERIFIED] GoalAlignmentScore composite computation correct");
}

#[test]
fn test_config_validation() {
    println!("\n============================================================");
    println!("TEST: test_config_validation");
    println!("============================================================");

    // Valid config
    let hierarchy = create_real_hierarchy();
    let config = AlignmentConfig::with_hierarchy(hierarchy);

    println!("\nVALID CONFIG:");
    let validation = config.validate();
    println!("  - validate(): {:?}", validation);
    assert!(
        validation.is_ok(),
        "FAIL: Valid config should pass validation"
    );

    // Invalid weights (sum != 1.0)
    // TASK-P0-001: Now only 3 levels (strategic, tactical, immediate)
    let invalid_config = AlignmentConfig {
        level_weights: LevelWeights {
            strategic: 0.5,
            tactical: 0.5,
            immediate: 0.5, // Sum = 1.5 != 1.0
        },
        ..AlignmentConfig::default()
    };

    println!("\nINVALID WEIGHTS CONFIG:");
    let validation = invalid_config.validate();
    println!("  - validate(): {:?}", validation);
    assert!(validation.is_err(), "FAIL: Invalid weights should fail");

    println!("\n[VERIFIED] Config validation works correctly");
}

#[test]
fn test_error_types_are_descriptive() {
    println!("\n============================================================");
    println!("TEST: test_error_types_are_descriptive");
    println!("============================================================");

    let errors = [
        AlignmentError::NoTopLevelGoals,
        AlignmentError::GoalNotFound(Uuid::new_v4()),
        AlignmentError::EmptyFingerprint,
        AlignmentError::DimensionMismatch {
            expected: 13,
            got: 10,
        },
        AlignmentError::InvalidConfig("test error".into()),
        AlignmentError::Timeout {
            elapsed_ms: 10,
            limit_ms: 5,
        },
        AlignmentError::InvalidHierarchy("orphan nodes".into()),
        AlignmentError::ComputationFailed("NaN detected".into()),
    ];

    println!("\nERROR MESSAGES:");
    for e in &errors {
        println!(
            "  - {}: {}",
            std::any::type_name_of_val(e)
                .split("::")
                .last()
                .unwrap_or("?"),
            e
        );
    }

    // Check recoverable
    println!("\nRECOVERABILITY:");
    for e in &errors {
        println!("  - {}: recoverable={}", e, e.is_recoverable());
    }

    // Only Timeout should be recoverable
    assert!(
        AlignmentError::Timeout {
            elapsed_ms: 10,
            limit_ms: 5
        }
        .is_recoverable(),
        "FAIL: Timeout should be recoverable"
    );
    assert!(
        !AlignmentError::NoTopLevelGoals.is_recoverable(),
        "FAIL: NoTopLevelGoals should not be recoverable"
    );

    println!("\n[VERIFIED] Error types are descriptive and categorized correctly");
}
