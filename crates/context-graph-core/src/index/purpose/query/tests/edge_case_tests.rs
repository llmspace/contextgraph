//! Edge case tests for purpose query types.

use uuid::Uuid;

use super::helpers::create_purpose_vector;
use crate::index::purpose::query::{PurposeQuery, PurposeQueryTarget};

#[test]
fn test_query_with_from_memory_target() {
    let id = Uuid::new_v4();
    let query = PurposeQuery::new(PurposeQueryTarget::from_memory(id), 5, 0.3).unwrap();

    assert!(query.target.requires_memory_lookup());
    assert_eq!(query.limit, 5);

    println!("[VERIFIED] PurposeQuery works with FromMemory target");
}

#[test]
fn test_query_with_pattern_target() {
    let target = PurposeQueryTarget::pattern(10, 0.8).unwrap();
    let query = PurposeQuery::new(target, 50, 0.0).unwrap();

    assert!(!query.target.requires_memory_lookup());

    match query.target {
        PurposeQueryTarget::Pattern {
            min_cluster_size,
            coherence_threshold,
        } => {
            assert_eq!(min_cluster_size, 10);
            assert!((coherence_threshold - 0.8).abs() < f32::EPSILON);
        }
        _ => panic!("Expected Pattern target"),
    }

    println!("[VERIFIED] PurposeQuery works with Pattern target");
}

#[test]
fn test_large_limit_value() {
    let pv = create_purpose_vector(0.5);
    let query = PurposeQuery::new(PurposeQueryTarget::Vector(pv), 1_000_000, 0.0).unwrap();

    assert_eq!(query.limit, 1_000_000);

    println!("[VERIFIED] PurposeQuery accepts large limit values");
}
