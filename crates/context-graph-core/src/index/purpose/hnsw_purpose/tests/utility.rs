//! Utility and edge case tests for HnswPurposeIndex.

use uuid::Uuid;

use crate::index::config::{DistanceMetric, HnswConfig, PURPOSE_VECTOR_DIM};
use crate::types::fingerprint::PurposeVector;

use crate::index::purpose::entry::{GoalId, PurposeIndexEntry, PurposeMetadata};
use crate::index::purpose::hnsw_purpose::{HnswPurposeIndex, PurposeIndexOps};
use crate::index::purpose::query::{PurposeQuery, PurposeQueryTarget};

// =========================================================================
// Helper functions (duplicated for test module isolation)
// =========================================================================

fn create_purpose_vector(base: f32, variation: f32) -> PurposeVector {
    let mut alignments = [0.0f32; PURPOSE_VECTOR_DIM];
    for (i, alignment) in alignments.iter_mut().enumerate() {
        *alignment = (base + (i as f32 * variation)).clamp(0.0, 1.0);
    }
    PurposeVector::new(alignments)
}

fn create_metadata(goal: &str) -> PurposeMetadata {
    PurposeMetadata::new(GoalId::new(goal), 0.85).unwrap()
}

fn create_entry(base: f32, goal: &str) -> PurposeIndexEntry {
    let pv = create_purpose_vector(base, 0.02);
    let metadata = create_metadata(goal);
    PurposeIndexEntry::new(Uuid::new_v4(), pv, metadata)
}

fn purpose_config() -> HnswConfig {
    HnswConfig::new(16, 200, 100, DistanceMetric::Cosine, PURPOSE_VECTOR_DIM)
}

// =========================================================================
// Utility Method Tests
// =========================================================================

#[test]
fn test_contains() {
    let mut index = HnswPurposeIndex::new(purpose_config()).unwrap();
    let entry = create_entry(0.5, "goal");
    let id = entry.memory_id;
    let other_id = Uuid::new_v4();

    assert!(!index.contains(id));
    assert!(!index.contains(other_id));

    index.insert(entry).unwrap();

    assert!(index.contains(id));
    assert!(!index.contains(other_id));

    println!("[VERIFIED] contains returns correct status");
}

#[test]
fn test_len_and_is_empty() {
    let mut index = HnswPurposeIndex::new(purpose_config()).unwrap();

    assert!(index.is_empty());
    assert_eq!(index.len(), 0);

    index.insert(create_entry(0.5, "goal")).unwrap();

    assert!(!index.is_empty());
    assert_eq!(index.len(), 1);

    println!("[VERIFIED] len and is_empty work correctly");
}

#[test]
fn test_clear() {
    let mut index = HnswPurposeIndex::new(purpose_config()).unwrap();

    for i in 0..10 {
        index
            .insert(create_entry(0.3 + i as f32 * 0.05, &format!("goal_{}", i % 3)))
            .unwrap();
    }

    assert_eq!(index.len(), 10);
    assert!(index.goal_count() > 0);

    index.clear();

    assert!(index.is_empty());
    assert_eq!(index.len(), 0);
    assert_eq!(index.goal_count(), 0);

    println!("[VERIFIED] clear removes all entries and indexes");
}

#[test]
fn test_goals_returns_all_goals() {
    let mut index = HnswPurposeIndex::new(purpose_config()).unwrap();

    index.insert(create_entry(0.5, "alpha")).unwrap();
    index.insert(create_entry(0.6, "beta")).unwrap();
    index.insert(create_entry(0.7, "gamma")).unwrap();

    let goals = index.goals();

    assert_eq!(goals.len(), 3);

    let goal_strs: Vec<&str> = goals.iter().map(|g| g.as_str()).collect();
    assert!(goal_strs.contains(&"alpha"));
    assert!(goal_strs.contains(&"beta"));
    assert!(goal_strs.contains(&"gamma"));

    println!("[VERIFIED] goals returns all distinct goals");
}

// =========================================================================
// Edge Case Tests
// =========================================================================

#[test]
fn test_filter_returns_empty_when_no_matches() {
    let mut index = HnswPurposeIndex::new(purpose_config()).unwrap();

    index.insert(create_entry(0.5, "goal_a")).unwrap();

    // Search with non-existent goal filter
    let query = PurposeQuery::new(
        PurposeQueryTarget::vector(create_purpose_vector(0.5, 0.02)),
        10,
        0.0,
    )
    .unwrap()
    .with_goal_filter(GoalId::new("non_existent_goal"));

    let results = index.search(&query).unwrap();

    assert!(results.is_empty());

    println!("[VERIFIED] Filter returns empty when no matches exist");
}

#[test]
fn test_search_results_sorted_by_similarity() {
    let mut index = HnswPurposeIndex::new(purpose_config()).unwrap();

    for i in 0..10 {
        index
            .insert(create_entry(0.1 * i as f32, "goal"))
            .unwrap();
    }

    let query = PurposeQuery::new(
        PurposeQueryTarget::vector(create_purpose_vector(0.5, 0.01)),
        10,
        0.0,
    )
    .unwrap();

    let results = index.search(&query).unwrap();

    // Verify descending similarity order
    for i in 1..results.len() {
        assert!(
            results[i - 1].purpose_similarity >= results[i].purpose_similarity,
            "Results not sorted: {} should be >= {}",
            results[i - 1].purpose_similarity,
            results[i].purpose_similarity
        );
    }

    println!("[VERIFIED] Search results are sorted by similarity descending");
}

#[test]
fn test_search_result_contains_complete_data() {
    let mut index = HnswPurposeIndex::new(purpose_config()).unwrap();

    let entry = create_entry(0.75, "complete_goal");
    index.insert(entry.clone()).unwrap();

    let query = PurposeQuery::new(
        PurposeQueryTarget::vector(entry.purpose_vector.clone()),
        1,
        0.0,
    )
    .unwrap();

    let results = index.search(&query).unwrap();

    assert_eq!(results.len(), 1);
    let result = &results[0];

    assert_eq!(result.memory_id, entry.memory_id);
    assert_eq!(
        result.purpose_vector.alignments,
        entry.purpose_vector.alignments
    );
    assert_eq!(
        result.metadata.primary_goal.as_str(),
        entry.metadata.primary_goal.as_str()
    );

    println!("[VERIFIED] Search result contains complete entry data");
}
