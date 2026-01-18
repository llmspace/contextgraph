//! Tests for PurposeQuery.

use super::helpers::create_purpose_vector;
use crate::index::purpose::entry::GoalId;
use crate::index::purpose::query::{PurposeQuery, PurposeQueryTarget};

#[test]
fn test_purpose_query_new_valid() {
    let pv = create_purpose_vector(0.5);
    let query = PurposeQuery::new(PurposeQueryTarget::Vector(pv), 10, 0.7).unwrap();

    assert_eq!(query.limit, 10);
    assert!((query.min_similarity - 0.7).abs() < f32::EPSILON);
    assert!(query.goal_filter.is_none());

    println!("[VERIFIED] PurposeQuery::new creates valid query");
}

#[test]
fn test_purpose_query_new_boundary_min_similarity() {
    let pv = create_purpose_vector(0.5);

    // Test min_similarity = 0.0
    let query = PurposeQuery::new(PurposeQueryTarget::Vector(pv.clone()), 10, 0.0).unwrap();
    assert_eq!(query.min_similarity, 0.0);

    // Test min_similarity = 1.0
    let query = PurposeQuery::new(PurposeQueryTarget::Vector(pv), 10, 1.0).unwrap();
    assert_eq!(query.min_similarity, 1.0);

    println!("[VERIFIED] PurposeQuery::new accepts min_similarity boundary values 0.0 and 1.0");
}

#[test]
fn test_purpose_query_new_invalid_limit_zero() {
    let pv = create_purpose_vector(0.5);
    let result = PurposeQuery::new(PurposeQueryTarget::Vector(pv), 0, 0.5);

    assert!(result.is_err());
    let err = result.unwrap_err();
    let msg = err.to_string();
    assert!(msg.contains("limit"));

    println!(
        "[VERIFIED] FAIL FAST: PurposeQuery::new rejects limit=0: {}",
        msg
    );
}

#[test]
fn test_purpose_query_new_invalid_min_similarity_over() {
    let pv = create_purpose_vector(0.5);
    let result = PurposeQuery::new(PurposeQueryTarget::Vector(pv), 10, 1.5);

    assert!(result.is_err());
    let err = result.unwrap_err();
    let msg = err.to_string();
    assert!(msg.contains("min_similarity"));
    assert!(msg.contains("1.5"));

    println!(
        "[VERIFIED] FAIL FAST: PurposeQuery::new rejects min_similarity=1.5: {}",
        msg
    );
}

#[test]
fn test_purpose_query_new_invalid_min_similarity_under() {
    let pv = create_purpose_vector(0.5);
    let result = PurposeQuery::new(PurposeQueryTarget::Vector(pv), 10, -0.1);

    assert!(result.is_err());
    let err = result.unwrap_err();
    let msg = err.to_string();
    assert!(msg.contains("min_similarity"));

    println!(
        "[VERIFIED] FAIL FAST: PurposeQuery::new rejects min_similarity=-0.1: {}",
        msg
    );
}

#[test]
fn test_purpose_query_new_invalid_min_similarity_nan() {
    let pv = create_purpose_vector(0.5);
    let result = PurposeQuery::new(PurposeQueryTarget::Vector(pv), 10, f32::NAN);

    assert!(result.is_err());
    let err = result.unwrap_err();
    let msg = err.to_string();
    assert!(msg.contains("NaN") || msg.contains("min_similarity"));

    println!(
        "[VERIFIED] FAIL FAST: PurposeQuery::new rejects min_similarity=NaN: {}",
        msg
    );
}

#[test]
fn test_purpose_query_with_goal_filter() {
    let pv = create_purpose_vector(0.5);
    let query = PurposeQuery::new(PurposeQueryTarget::Vector(pv), 10, 0.5)
        .unwrap()
        .with_goal_filter(GoalId::new("master_ml"));

    assert!(query.goal_filter.is_some());
    assert_eq!(query.goal_filter.as_ref().unwrap().as_str(), "master_ml");

    println!("[VERIFIED] PurposeQuery::with_goal_filter sets goal filter correctly");
}

#[test]
fn test_purpose_query_validate() {
    let pv = create_purpose_vector(0.5);
    let query = PurposeQuery::new(PurposeQueryTarget::Vector(pv), 10, 0.5).unwrap();

    assert!(query.validate().is_ok());

    println!("[VERIFIED] PurposeQuery::validate passes for valid query");
}

#[test]
fn test_purpose_query_has_filters() {
    let pv = create_purpose_vector(0.5);

    // No filters
    let query = PurposeQuery::new(PurposeQueryTarget::Vector(pv.clone()), 10, 0.5).unwrap();
    assert!(!query.has_filters());
    assert_eq!(query.filter_count(), 0);

    // Goal filter only
    let query = query.with_goal_filter(GoalId::new("test"));
    assert!(query.has_filters());
    assert_eq!(query.filter_count(), 1);

    println!("[VERIFIED] has_filters and filter_count work correctly");
}
