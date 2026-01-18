//! Tests for PurposeQueryBuilder.

use super::helpers::create_purpose_vector;
use crate::index::purpose::entry::GoalId;
use crate::index::purpose::query::{PurposeQuery, PurposeQueryBuilder, PurposeQueryTarget};

#[test]
fn test_purpose_query_builder_full() {
    let pv = create_purpose_vector(0.5);

    let query = PurposeQueryBuilder::new()
        .target(PurposeQueryTarget::Vector(pv))
        .limit(20)
        .min_similarity(0.8)
        .goal_filter(GoalId::new("learn_pytorch"))
        .build()
        .unwrap();

    assert_eq!(query.limit, 20);
    assert!((query.min_similarity - 0.8).abs() < f32::EPSILON);
    assert_eq!(
        query.goal_filter.as_ref().unwrap().as_str(),
        "learn_pytorch"
    );

    println!("[VERIFIED] PurposeQueryBuilder builds complete query with all fields");
}

#[test]
fn test_purpose_query_builder_minimal() {
    let pv = create_purpose_vector(0.5);

    let query = PurposeQueryBuilder::new()
        .target(PurposeQueryTarget::Vector(pv))
        .limit(5)
        .min_similarity(0.0)
        .build()
        .unwrap();

    assert_eq!(query.limit, 5);
    assert_eq!(query.min_similarity, 0.0);
    assert!(query.goal_filter.is_none());

    println!("[VERIFIED] PurposeQueryBuilder builds minimal query without filters");
}

#[test]
fn test_purpose_query_builder_missing_target() {
    let result = PurposeQueryBuilder::new()
        .limit(10)
        .min_similarity(0.5)
        .build();

    assert!(result.is_err());
    let err = result.unwrap_err();
    let msg = err.to_string();
    assert!(msg.contains("target"));

    println!(
        "[VERIFIED] FAIL FAST: PurposeQueryBuilder::build rejects missing target: {}",
        msg
    );
}

#[test]
fn test_purpose_query_builder_missing_limit() {
    let pv = create_purpose_vector(0.5);

    let result = PurposeQueryBuilder::new()
        .target(PurposeQueryTarget::Vector(pv))
        .min_similarity(0.5)
        .build();

    assert!(result.is_err());
    let err = result.unwrap_err();
    let msg = err.to_string();
    assert!(msg.contains("limit"));

    println!(
        "[VERIFIED] FAIL FAST: PurposeQueryBuilder::build rejects missing limit: {}",
        msg
    );
}

#[test]
fn test_purpose_query_builder_missing_min_similarity() {
    let pv = create_purpose_vector(0.5);

    let result = PurposeQueryBuilder::new()
        .target(PurposeQueryTarget::Vector(pv))
        .limit(10)
        .build();

    assert!(result.is_err());
    let err = result.unwrap_err();
    let msg = err.to_string();
    assert!(msg.contains("min_similarity"));

    println!(
        "[VERIFIED] FAIL FAST: PurposeQueryBuilder::build rejects missing min_similarity: {}",
        msg
    );
}

#[test]
fn test_purpose_query_builder_chained() {
    let pv = create_purpose_vector(0.7);

    // Test that builder methods can be chained in any order
    let query = PurposeQuery::builder()
        .min_similarity(0.6)
        .limit(15)
        .target(PurposeQueryTarget::Vector(pv))
        .goal_filter(GoalId::new("frontier"))
        .build()
        .unwrap();

    assert_eq!(query.limit, 15);
    assert!((query.min_similarity - 0.6).abs() < f32::EPSILON);

    println!("[VERIFIED] PurposeQueryBuilder allows chaining in any order");
}
