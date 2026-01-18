//! Query Tests - PurposeQuery builder and validation

use super::helpers::create_purpose_vector;
use crate::index::purpose::entry::GoalId;
use crate::index::purpose::query::{PurposeQuery, PurposeQueryTarget};
use uuid::Uuid;

#[test]
fn test_query_builder_with_validation() {
    let pv = create_purpose_vector(0.7, 0.02);

    let query = PurposeQuery::builder()
        .target(PurposeQueryTarget::Vector(pv))
        .limit(10)
        .min_similarity(0.5)
        .build()
        .unwrap();

    assert_eq!(query.limit, 10);
    assert!((query.min_similarity - 0.5).abs() < f32::EPSILON);

    println!("[VERIFIED] PurposeQuery builder creates valid query");
}

#[test]
fn test_query_all_target_variants() {
    let pv = create_purpose_vector(0.5, 0.02);

    // Vector target
    let target = PurposeQueryTarget::vector(pv.clone());
    assert!(!target.requires_memory_lookup());

    // Pattern target
    let target = PurposeQueryTarget::pattern(5, 0.7).unwrap();
    assert!(!target.requires_memory_lookup());

    // FromMemory target
    let target = PurposeQueryTarget::from_memory(Uuid::new_v4());
    assert!(target.requires_memory_lookup());

    println!("[VERIFIED] All PurposeQueryTarget variants work correctly");
}

#[test]
fn test_query_filter_combinations() {
    let pv = create_purpose_vector(0.5, 0.02);

    // No filters
    let query = PurposeQuery::new(PurposeQueryTarget::Vector(pv.clone()), 10, 0.0).unwrap();
    assert!(!query.has_filters());
    assert_eq!(query.filter_count(), 0);

    // Goal filter only
    let query = query.with_goal_filter(GoalId::new("test_goal"));
    assert!(query.has_filters());
    assert_eq!(query.filter_count(), 1);

    println!("[VERIFIED] Query filter combinations work correctly");
}

#[test]
fn test_query_rejects_invalid_min_similarity_out_of_range() {
    let pv = create_purpose_vector(0.5, 0.02);

    // Over 1.0
    let result = PurposeQuery::new(PurposeQueryTarget::Vector(pv.clone()), 10, 1.5);
    assert!(result.is_err());
    let msg = result.unwrap_err().to_string();
    assert!(msg.contains("min_similarity"));

    // Under 0.0
    let result = PurposeQuery::new(PurposeQueryTarget::Vector(pv.clone()), 10, -0.1);
    assert!(result.is_err());

    println!(
        "[VERIFIED] FAIL FAST: Query rejects invalid min_similarity: {}",
        msg
    );
}

#[test]
fn test_query_rejects_limit_zero() {
    let pv = create_purpose_vector(0.5, 0.02);

    let result = PurposeQuery::new(PurposeQueryTarget::Vector(pv), 0, 0.5);
    assert!(result.is_err());
    let msg = result.unwrap_err().to_string();
    assert!(msg.contains("limit"));

    println!("[VERIFIED] FAIL FAST: Query rejects limit=0: {}", msg);
}

#[test]
fn test_query_pattern_rejects_invalid_coherence() {
    // Over 1.0
    let result = PurposeQueryTarget::pattern(5, 1.5);
    assert!(result.is_err());

    // Under 0.0
    let result = PurposeQueryTarget::pattern(5, -0.1);
    assert!(result.is_err());

    println!("[VERIFIED] FAIL FAST: Pattern target rejects invalid coherence_threshold");
}

#[test]
fn test_query_pattern_rejects_zero_cluster_size() {
    let result = PurposeQueryTarget::pattern(0, 0.7);
    assert!(result.is_err());
    let msg = result.unwrap_err().to_string();
    assert!(msg.contains("min_cluster_size"));

    println!(
        "[VERIFIED] FAIL FAST: Pattern target rejects min_cluster_size=0: {}",
        msg
    );
}
