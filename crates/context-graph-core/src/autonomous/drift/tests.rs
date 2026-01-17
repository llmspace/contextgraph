//! Tests for teleological drift detection.

use chrono::Utc;

use crate::teleological::{SearchStrategy, TeleologicalComparator};
use crate::types::fingerprint::SparseVector;
use crate::types::SemanticFingerprint;

use super::detector::TeleologicalDriftDetector;
use super::error::DriftError;
use super::history::{DriftHistory, DriftHistoryEntry};
use super::types::{DriftLevel, DriftThresholds, DriftTrend};

// ============================================
// TEST FIXTURE HELPERS
// ============================================

/// Create a test goal with aligned values.
fn create_test_goal() -> SemanticFingerprint {
    create_fingerprint_with_value(1.0)
}

/// Create a test memory with high similarity to the goal.
fn create_test_memory() -> SemanticFingerprint {
    create_fingerprint_with_value(1.0)
}

/// Create a memory with specified drift (lower similarity).
fn create_drifted_memory(base_similarity: f32) -> SemanticFingerprint {
    // Create a fingerprint that will produce approximately base_similarity
    // when compared to the test goal
    create_fingerprint_with_value(base_similarity)
}

/// Create a heavily drifted memory.
fn create_heavily_drifted_memory() -> SemanticFingerprint {
    create_drifted_memory(0.3)
}

/// Create a goal with NaN values.
fn create_goal_with_nan() -> SemanticFingerprint {
    let mut fp = create_test_goal();
    if !fp.e1_semantic.is_empty() {
        fp.e1_semantic[0] = f32::NAN;
    }
    fp
}

/// Create a goal with Inf values.
fn create_goal_with_inf() -> SemanticFingerprint {
    let mut fp = create_test_goal();
    if !fp.e1_semantic.is_empty() {
        fp.e1_semantic[0] = f32::INFINITY;
    }
    fp
}

/// Create a normalized fingerprint with given base value.
fn create_fingerprint_with_value(val: f32) -> SemanticFingerprint {
    use crate::types::fingerprint::{
        E10_DIM, E11_DIM, E1_DIM, E2_DIM, E3_DIM, E4_DIM, E5_DIM, E7_DIM, E8_DIM, E9_DIM,
    };

    let create_normalized_vec = |dim: usize, v: f32| -> Vec<f32> {
        let mut vec = vec![v; dim];
        let norm: f32 = vec.iter().map(|x| x * x).sum::<f32>().sqrt();
        if norm > f32::EPSILON {
            for x in &mut vec {
                *x /= norm;
            }
        }
        vec
    };

    SemanticFingerprint {
        e1_semantic: create_normalized_vec(E1_DIM, val),
        e2_temporal_recent: create_normalized_vec(E2_DIM, val),
        e3_temporal_periodic: create_normalized_vec(E3_DIM, val),
        e4_temporal_positional: create_normalized_vec(E4_DIM, val),
        e5_causal: create_normalized_vec(E5_DIM, val),
        e6_sparse: SparseVector::empty(),
        e7_code: create_normalized_vec(E7_DIM, val),
        e8_graph: create_normalized_vec(E8_DIM, val),
        e9_hdc: create_normalized_vec(E9_DIM, val),
        e10_multimodal: create_normalized_vec(E10_DIM, val),
        e11_entity: create_normalized_vec(E11_DIM, val),
        e12_late_interaction: vec![vec![val / 128.0_f32.sqrt(); 128]],
        e13_splade: SparseVector::empty(),
    }
}

// ============================================
// DRIFT LEVEL CLASSIFICATION TESTS
// ============================================

#[test]
fn test_drift_level_from_similarity_none() {
    let thresholds = DriftThresholds::default();
    assert_eq!(
        DriftLevel::from_similarity(0.90, &thresholds),
        DriftLevel::None
    );
    assert_eq!(
        DriftLevel::from_similarity(0.85, &thresholds),
        DriftLevel::None
    );
}

#[test]
fn test_drift_level_from_similarity_low() {
    let thresholds = DriftThresholds::default();
    assert_eq!(
        DriftLevel::from_similarity(0.84, &thresholds),
        DriftLevel::Low
    );
    assert_eq!(
        DriftLevel::from_similarity(0.70, &thresholds),
        DriftLevel::Low
    );
}

#[test]
fn test_drift_level_from_similarity_medium() {
    let thresholds = DriftThresholds::default();
    assert_eq!(
        DriftLevel::from_similarity(0.69, &thresholds),
        DriftLevel::Medium
    );
    assert_eq!(
        DriftLevel::from_similarity(0.55, &thresholds),
        DriftLevel::Medium
    );
}

#[test]
fn test_drift_level_from_similarity_high() {
    let thresholds = DriftThresholds::default();
    assert_eq!(
        DriftLevel::from_similarity(0.54, &thresholds),
        DriftLevel::High
    );
    assert_eq!(
        DriftLevel::from_similarity(0.40, &thresholds),
        DriftLevel::High
    );
}

#[test]
fn test_drift_level_from_similarity_critical() {
    let thresholds = DriftThresholds::default();
    assert_eq!(
        DriftLevel::from_similarity(0.39, &thresholds),
        DriftLevel::Critical
    );
    assert_eq!(
        DriftLevel::from_similarity(0.0, &thresholds),
        DriftLevel::Critical
    );
}

#[test]
fn test_drift_level_ordering() {
    // Critical < High < Medium < Low < None (for sorting worst first)
    assert!(DriftLevel::Critical < DriftLevel::High);
    assert!(DriftLevel::High < DriftLevel::Medium);
    assert!(DriftLevel::Medium < DriftLevel::Low);
    assert!(DriftLevel::Low < DriftLevel::None);
}

// ============================================
// FAIL FAST TESTS
// ============================================

#[test]
fn test_fail_fast_empty_memories() {
    let comparator = TeleologicalComparator::new();
    let detector = TeleologicalDriftDetector::new(comparator);
    let goal = create_test_goal();

    let result = detector.check_drift(&[], &goal, SearchStrategy::Cosine);

    assert!(matches!(result, Err(DriftError::EmptyMemories)));
}

#[test]
fn test_fail_fast_invalid_goal_nan() {
    let comparator = TeleologicalComparator::new();
    let detector = TeleologicalDriftDetector::new(comparator);
    let goal = create_goal_with_nan();
    let memories = vec![create_test_memory()];

    let result = detector.check_drift(&memories, &goal, SearchStrategy::Cosine);

    assert!(matches!(result, Err(DriftError::InvalidGoal { .. })));
}

#[test]
fn test_fail_fast_invalid_goal_inf() {
    let comparator = TeleologicalComparator::new();
    let detector = TeleologicalDriftDetector::new(comparator);
    let goal = create_goal_with_inf();
    let memories = vec![create_test_memory()];

    let result = detector.check_drift(&memories, &goal, SearchStrategy::Cosine);

    assert!(matches!(result, Err(DriftError::InvalidGoal { .. })));
}

// ============================================
// PER-EMBEDDER ANALYSIS TESTS
// ============================================

#[test]
fn test_per_embedder_breakdown_all_13() {
    let comparator = TeleologicalComparator::new();
    let detector = TeleologicalDriftDetector::new(comparator);
    let goal = create_test_goal();
    let memories = vec![create_test_memory()];

    let result = detector
        .check_drift(&memories, &goal, SearchStrategy::Cosine)
        .unwrap();

    // Must have exactly 13 embedder entries
    assert_eq!(result.per_embedder_drift.embedder_drift.len(), 13);

    // Each embedder must be present
    for embedder in crate::teleological::Embedder::all() {
        let found = result
            .per_embedder_drift
            .embedder_drift
            .iter()
            .any(|e| e.embedder == embedder);
        assert!(found, "Missing embedder: {:?}", embedder);
    }
}

#[test]
fn test_per_embedder_similarity_valid_range() {
    let comparator = TeleologicalComparator::new();
    let detector = TeleologicalDriftDetector::new(comparator);
    let goal = create_test_goal();
    let memories = vec![create_test_memory()];

    let result = detector
        .check_drift(&memories, &goal, SearchStrategy::Cosine)
        .unwrap();

    for info in &result.per_embedder_drift.embedder_drift {
        assert!(
            info.similarity >= 0.0,
            "Similarity < 0 for {:?}",
            info.embedder
        );
        assert!(
            info.similarity <= 1.0,
            "Similarity > 1 for {:?}",
            info.embedder
        );
        assert!(
            !info.similarity.is_nan(),
            "NaN similarity for {:?}",
            info.embedder
        );
    }
}

#[test]
fn test_drift_score_equals_one_minus_similarity() {
    let comparator = TeleologicalComparator::new();
    let detector = TeleologicalDriftDetector::new(comparator);
    let goal = create_test_goal();
    let memories = vec![create_test_memory()];

    let result = detector
        .check_drift(&memories, &goal, SearchStrategy::Cosine)
        .unwrap();

    let expected_drift = 1.0 - result.overall_drift.similarity;
    assert!(
        (result.overall_drift.drift_score - expected_drift).abs() < 0.0001,
        "Drift score {} != 1 - similarity {}",
        result.overall_drift.drift_score,
        expected_drift
    );
}

// ============================================
// MOST DRIFTED EMBEDDERS TESTS
// ============================================

#[test]
fn test_most_drifted_sorted_worst_first() {
    let comparator = TeleologicalComparator::new();
    let detector = TeleologicalDriftDetector::new(comparator);
    let goal = create_test_goal();
    let memories = vec![create_drifted_memory(0.6)];

    let result = detector
        .check_drift(&memories, &goal, SearchStrategy::Cosine)
        .unwrap();

    // Verify descending order by drift level (worst first)
    for window in result.most_drifted_embedders.windows(2) {
        assert!(
            window[0].drift_level <= window[1].drift_level,
            "Not sorted: {:?} should come before {:?}",
            window[0].drift_level,
            window[1].drift_level
        );
    }
}

#[test]
fn test_most_drifted_max_five() {
    let comparator = TeleologicalComparator::new();
    let detector = TeleologicalDriftDetector::new(comparator);
    let goal = create_test_goal();
    let memories = vec![create_heavily_drifted_memory()];

    let result = detector
        .check_drift(&memories, &goal, SearchStrategy::Cosine)
        .unwrap();

    assert!(
        result.most_drifted_embedders.len() <= 5,
        "Should return at most 5 drifted embedders"
    );
}

// ============================================
// TREND ANALYSIS TESTS
// ============================================

#[test]
fn test_trend_requires_minimum_samples() {
    let comparator = TeleologicalComparator::new();
    let mut detector = TeleologicalDriftDetector::new(comparator);
    let goal = create_test_goal();
    let memories = vec![create_test_memory()];

    // Add only 2 samples
    for _ in 0..2 {
        let _ =
            detector.check_drift_with_history(&memories, &goal, "goal-1", SearchStrategy::Cosine);
    }

    let trend = detector.get_trend("goal-1");
    assert!(trend.is_none(), "Trend should require >= 3 samples");
}

#[test]
fn test_trend_available_with_enough_samples() {
    let comparator = TeleologicalComparator::new();
    let mut detector = TeleologicalDriftDetector::new(comparator);
    let goal = create_test_goal();
    let memories = vec![create_test_memory()];

    // Add 5 samples
    for _ in 0..5 {
        let _ =
            detector.check_drift_with_history(&memories, &goal, "goal-1", SearchStrategy::Cosine);
    }

    let trend = detector.get_trend("goal-1");
    assert!(trend.is_some(), "Trend should be available with 5 samples");
    assert_eq!(trend.unwrap().samples, 5);
}

#[test]
fn test_trend_direction_stable_for_identical() {
    let comparator = TeleologicalComparator::new();
    let mut detector = TeleologicalDriftDetector::new(comparator);
    let goal = create_test_goal();
    let memories = vec![create_test_memory()];

    // Add identical samples (should be stable)
    for _ in 0..5 {
        let _ =
            detector.check_drift_with_history(&memories, &goal, "goal-1", SearchStrategy::Cosine);
    }

    let trend = detector.get_trend("goal-1").unwrap();
    assert_eq!(
        trend.direction,
        DriftTrend::Stable,
        "Identical samples should show stable trend"
    );
}

// ============================================
// RECOMMENDATIONS TESTS
// ============================================

#[test]
fn test_recommendations_only_for_medium_plus() {
    let comparator = TeleologicalComparator::new();
    let detector = TeleologicalDriftDetector::new(comparator);
    let goal = create_test_goal();
    let memories = vec![create_test_memory()];

    let result = detector
        .check_drift(&memories, &goal, SearchStrategy::Cosine)
        .unwrap();

    for rec in &result.recommendations {
        // Find the corresponding embedder info
        let info = result
            .per_embedder_drift
            .embedder_drift
            .iter()
            .find(|e| e.embedder == rec.embedder)
            .unwrap();

        assert!(
            info.drift_level.needs_recommendation(),
            "Recommendation for {:?} with level {:?} (should be Medium or worse)",
            rec.embedder,
            info.drift_level
        );
    }
}

#[test]
fn test_recommendations_priority_matches_drift_level() {
    let comparator = TeleologicalComparator::new();
    let detector = TeleologicalDriftDetector::new(comparator);
    let goal = create_test_goal();
    let memories = vec![create_heavily_drifted_memory()];

    let result = detector
        .check_drift(&memories, &goal, SearchStrategy::Cosine)
        .unwrap();

    for rec in &result.recommendations {
        let info = result
            .per_embedder_drift
            .embedder_drift
            .iter()
            .find(|e| e.embedder == rec.embedder)
            .unwrap();

        let expected_priority =
            super::recommendations::RecommendationPriority::from(info.drift_level);
        assert_eq!(
            rec.priority, expected_priority,
            "Priority mismatch for {:?}",
            rec.embedder
        );
    }
}

// ============================================
// HISTORY TESTS
// ============================================

#[test]
fn test_history_per_goal_isolation() {
    let comparator = TeleologicalComparator::new();
    let mut detector = TeleologicalDriftDetector::new(comparator);
    let goal1 = create_test_goal();
    let goal2 = create_test_goal();
    let memories = vec![create_test_memory()];

    // Add to goal-1
    for _ in 0..3 {
        let _ =
            detector.check_drift_with_history(&memories, &goal1, "goal-1", SearchStrategy::Cosine);
    }

    // Add to goal-2
    for _ in 0..5 {
        let _ =
            detector.check_drift_with_history(&memories, &goal2, "goal-2", SearchStrategy::Cosine);
    }

    let trend1 = detector.get_trend("goal-1");
    let trend2 = detector.get_trend("goal-2");

    assert_eq!(trend1.unwrap().samples, 3);
    assert_eq!(trend2.unwrap().samples, 5);
}

#[test]
fn test_history_entry_has_per_embedder_array() {
    let mut history = DriftHistory::new(100);

    let entry = DriftHistoryEntry {
        timestamp: Utc::now(),
        overall_similarity: 0.75,
        per_embedder: [
            0.8, 0.7, 0.6, 0.5, 0.9, 0.85, 0.75, 0.65, 0.55, 0.45, 0.95, 0.88, 0.72,
        ],
        memories_analyzed: 10,
    };

    history.add("test-goal", entry);

    let entries = history.get("test-goal").unwrap();
    assert_eq!(entries.len(), 1);
    assert_eq!(entries[0].per_embedder.len(), 13);
}

// ============================================
// CUSTOM THRESHOLDS TESTS
// ============================================

#[test]
fn test_custom_thresholds() {
    let custom = DriftThresholds {
        none_min: 0.90,
        low_min: 0.80,
        medium_min: 0.70,
        high_min: 0.60,
    };

    // With custom thresholds, 0.85 is Low (not None)
    assert_eq!(DriftLevel::from_similarity(0.85, &custom), DriftLevel::Low);

    // With default, 0.85 is None
    assert_eq!(
        DriftLevel::from_similarity(0.85, &DriftThresholds::default()),
        DriftLevel::None
    );
}

#[test]
fn test_invalid_thresholds_rejected() {
    let invalid = DriftThresholds {
        none_min: 0.70, // Less than low_min!
        low_min: 0.80,
        medium_min: 0.55,
        high_min: 0.40,
    };

    assert!(invalid.validate().is_err());
}

// ============================================
// SINGLE MEMORY TEST
// ============================================

#[test]
fn test_single_memory_analysis() {
    let comparator = TeleologicalComparator::new();
    let detector = TeleologicalDriftDetector::new(comparator);
    let goal = create_test_goal();
    let memories = vec![create_test_memory()];

    let result = detector
        .check_drift(&memories, &goal, SearchStrategy::Cosine)
        .unwrap();

    assert_eq!(result.analyzed_count, 1);
}

// ============================================
// TIMESTAMP TESTS
// ============================================

#[test]
fn test_result_has_recent_timestamp() {
    let comparator = TeleologicalComparator::new();
    let detector = TeleologicalDriftDetector::new(comparator);
    let goal = create_test_goal();
    let memories = vec![create_test_memory()];

    let before = Utc::now();
    let result = detector
        .check_drift(&memories, &goal, SearchStrategy::Cosine)
        .unwrap();
    let after = Utc::now();

    assert!(result.timestamp >= before);
    assert!(result.timestamp <= after);
}
