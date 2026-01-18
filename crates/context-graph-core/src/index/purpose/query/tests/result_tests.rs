//! Tests for PurposeSearchResult.

use uuid::Uuid;

use super::helpers::{create_metadata, create_purpose_vector};
use crate::index::config::PURPOSE_VECTOR_DIM;
use crate::index::purpose::entry::GoalId;
use crate::index::purpose::query::PurposeSearchResult;
use crate::types::fingerprint::PurposeVector;

#[test]
fn test_purpose_search_result_new() {
    let id = Uuid::new_v4();
    let pv = create_purpose_vector(0.8);
    let metadata = create_metadata("master_ml");

    let result = PurposeSearchResult::new(id, 0.95, pv.clone(), metadata);

    assert_eq!(result.memory_id, id);
    assert!((result.purpose_similarity - 0.95).abs() < f32::EPSILON);
    assert_eq!(result.purpose_vector.alignments, pv.alignments);
    assert_eq!(result.metadata.primary_goal.as_str(), "master_ml");

    println!("[VERIFIED] PurposeSearchResult::new creates result with all fields");
}

#[test]
fn test_purpose_search_result_aggregate_alignment() {
    let pv = PurposeVector::new([0.75; PURPOSE_VECTOR_DIM]);
    let metadata = create_metadata("test");
    let result = PurposeSearchResult::new(Uuid::new_v4(), 0.9, pv, metadata);

    let aggregate = result.aggregate_alignment();
    assert!((aggregate - 0.75).abs() < f32::EPSILON);

    println!("[VERIFIED] PurposeSearchResult::aggregate_alignment returns correct value");
}

#[test]
fn test_purpose_search_result_dominant_embedder() {
    let mut alignments = [0.5; PURPOSE_VECTOR_DIM];
    alignments[7] = 0.95; // E8 is dominant
    let pv = PurposeVector::new(alignments);
    let metadata = create_metadata("test");
    let result = PurposeSearchResult::new(Uuid::new_v4(), 0.9, pv, metadata);

    assert_eq!(result.dominant_embedder(), 7);

    println!("[VERIFIED] PurposeSearchResult::dominant_embedder returns correct index");
}

#[test]
fn test_purpose_search_result_coherence() {
    let pv = PurposeVector::new([0.8; PURPOSE_VECTOR_DIM]); // Uniform = high coherence
    let metadata = create_metadata("test");
    let result = PurposeSearchResult::new(Uuid::new_v4(), 0.9, pv, metadata);

    let coherence = result.coherence();
    assert!((coherence - 1.0).abs() < 1e-6);

    println!("[VERIFIED] PurposeSearchResult::coherence returns correct value");
}

#[test]
fn test_purpose_search_result_matches_goal() {
    let pv = create_purpose_vector(0.8);
    let metadata = create_metadata("master_ml");
    let result = PurposeSearchResult::new(Uuid::new_v4(), 0.9, pv, metadata);

    assert!(result.matches_goal(&GoalId::new("master_ml")));
    assert!(!result.matches_goal(&GoalId::new("other_goal")));

    println!("[VERIFIED] PurposeSearchResult::matches_goal filters correctly");
}

#[test]
fn test_purpose_search_result_clone() {
    let pv = create_purpose_vector(0.8);
    let metadata = create_metadata("test");
    let result = PurposeSearchResult::new(Uuid::new_v4(), 0.9, pv, metadata);

    let cloned = result.clone();

    assert_eq!(cloned.memory_id, result.memory_id);
    assert_eq!(cloned.purpose_similarity, result.purpose_similarity);
    assert_eq!(
        cloned.purpose_vector.alignments,
        result.purpose_vector.alignments
    );

    println!("[VERIFIED] PurposeSearchResult implements Clone correctly");
}

#[test]
fn test_purpose_search_result_debug() {
    let pv = create_purpose_vector(0.8);
    let metadata = create_metadata("test");
    let result = PurposeSearchResult::new(Uuid::nil(), 0.9, pv, metadata);

    let debug_str = format!("{:?}", result);

    assert!(debug_str.contains("PurposeSearchResult"));
    assert!(debug_str.contains("memory_id"));

    println!("[VERIFIED] PurposeSearchResult implements Debug correctly");
}
