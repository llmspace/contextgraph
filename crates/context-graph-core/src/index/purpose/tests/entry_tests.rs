//! Entry Tests - PurposeMetadata and PurposeIndexEntry

use super::helpers::{create_entry, create_metadata};
use crate::index::config::PURPOSE_VECTOR_DIM;
use crate::index::purpose::entry::{GoalId, PurposeIndexEntry, PurposeMetadata};
use crate::types::fingerprint::PurposeVector;
use std::time::Duration;
use uuid::Uuid;

#[test]
fn test_purpose_metadata_with_real_goal() {
    let goal = GoalId::new("master_machine_learning");

    let metadata = PurposeMetadata::new(goal.clone(), 0.85).unwrap();

    assert_eq!(metadata.primary_goal.as_str(), "master_machine_learning");
    assert!((metadata.confidence - 0.85).abs() < f32::EPSILON);

    // Verify timestamp is recent
    let elapsed = metadata.computed_at.elapsed().unwrap();
    assert!(elapsed < Duration::from_secs(1));

    println!("[VERIFIED] PurposeMetadata construction with real GoalId");
}

#[test]
fn test_purpose_metadata_valid_confidence_range() {
    // Test various valid confidence values
    for confidence in [0.0, 0.25, 0.5, 0.75, 1.0] {
        let metadata = PurposeMetadata::new(GoalId::new("test_goal"), confidence).unwrap();
        assert!((metadata.confidence - confidence).abs() < f32::EPSILON);
    }

    println!("[VERIFIED] PurposeMetadata works with all valid confidence values");
}

#[test]
fn test_purpose_index_entry_with_real_purpose_vector() {
    let alignments = [
        0.8, 0.7, 0.9, 0.6, 0.75, 0.65, 0.85, 0.72, 0.78, 0.68, 0.82, 0.71, 0.76,
    ];
    let pv = PurposeVector::new(alignments);
    let metadata = PurposeMetadata::new(GoalId::new("learn_pytorch"), 0.9).unwrap();

    let memory_id = Uuid::new_v4();
    let entry = PurposeIndexEntry::new(memory_id, pv, metadata);

    assert_eq!(entry.memory_id, memory_id);
    assert_eq!(entry.purpose_vector.alignments, alignments);
    assert_eq!(entry.metadata.primary_goal.as_str(), "learn_pytorch");

    println!("[VERIFIED] PurposeIndexEntry with real PurposeVector and metadata");
}

#[test]
fn test_entry_dimension_matches_purpose_vector_dim() {
    let entry = create_entry(0.5, "test");

    assert_eq!(entry.get_alignments().len(), PURPOSE_VECTOR_DIM);
    assert_eq!(entry.get_alignments().len(), 13);

    println!(
        "[VERIFIED] Entry alignments have dimension {}",
        PURPOSE_VECTOR_DIM
    );
}

#[test]
fn test_entry_aggregate_alignment_computed_correctly() {
    let uniform = PurposeVector::new([0.75; PURPOSE_VECTOR_DIM]);
    let metadata = create_metadata("test");
    let entry = PurposeIndexEntry::new(Uuid::new_v4(), uniform, metadata);

    let aggregate = entry.aggregate_alignment();
    assert!((aggregate - 0.75).abs() < f32::EPSILON);

    println!(
        "[VERIFIED] Entry aggregate_alignment returns correct mean: {:.4}",
        aggregate
    );
}

#[test]
fn test_entry_validation_rejects_invalid_confidence() {
    // Test confidence > 1.0
    let result = PurposeMetadata::new(GoalId::new("test"), 1.5);
    assert!(result.is_err());

    // Test confidence < 0.0
    let result = PurposeMetadata::new(GoalId::new("test"), -0.1);
    assert!(result.is_err());

    // Test NaN
    let result = PurposeMetadata::new(GoalId::new("test"), f32::NAN);
    assert!(result.is_err());

    println!("[VERIFIED] Entry validation detects invalid confidence values");
}
