//! Tests for HnswPurposeIndex.

mod search;
mod utility;

use uuid::Uuid;

use crate::index::config::{DistanceMetric, HnswConfig, PURPOSE_VECTOR_DIM};
use crate::types::fingerprint::PurposeVector;

use crate::index::purpose::entry::{GoalId, PurposeIndexEntry, PurposeMetadata};
use crate::index::purpose::hnsw_purpose::{HnswPurposeIndex, PurposeIndexOps};

// =========================================================================
// Helper functions for creating REAL test data (NO mocks)
// =========================================================================

/// Create a purpose vector with deterministic values.
fn create_purpose_vector(base: f32, variation: f32) -> PurposeVector {
    let mut alignments = [0.0f32; PURPOSE_VECTOR_DIM];
    for (i, alignment) in alignments.iter_mut().enumerate() {
        *alignment = (base + (i as f32 * variation)).clamp(0.0, 1.0);
    }
    PurposeVector::new(alignments)
}

/// Create metadata with given goal.
fn create_metadata(goal: &str) -> PurposeMetadata {
    PurposeMetadata::new(GoalId::new(goal), 0.85).unwrap()
}

/// Create a complete purpose index entry.
fn create_entry(base: f32, goal: &str) -> PurposeIndexEntry {
    let pv = create_purpose_vector(base, 0.02);
    let metadata = create_metadata(goal);
    PurposeIndexEntry::new(Uuid::new_v4(), pv, metadata)
}

/// Create a default HNSW config for purpose vectors.
fn purpose_config() -> HnswConfig {
    HnswConfig::new(16, 200, 100, DistanceMetric::Cosine, PURPOSE_VECTOR_DIM)
}

// =========================================================================
// Constructor Tests
// =========================================================================

#[test]
fn test_hnsw_purpose_index_new() {
    let config = purpose_config();
    let index = HnswPurposeIndex::new(config).unwrap();

    assert!(index.is_empty());
    assert_eq!(index.len(), 0);
    assert_eq!(index.goal_count(), 0);

    println!("[VERIFIED] HnswPurposeIndex::new creates empty index");
}

#[test]
fn test_hnsw_purpose_index_new_wrong_dimension() {
    let wrong_config = HnswConfig::new(16, 200, 100, DistanceMetric::Cosine, 100);
    let result = HnswPurposeIndex::new(wrong_config);

    assert!(result.is_err());
    let err = result.unwrap_err();
    let msg = err.to_string();
    assert!(msg.contains("13"));
    assert!(msg.contains("100"));

    println!(
        "[VERIFIED] FAIL FAST: HnswPurposeIndex::new rejects wrong dimension: {}",
        msg
    );
}

#[test]
fn test_hnsw_purpose_index_with_capacity() {
    let config = purpose_config();
    let index = HnswPurposeIndex::with_capacity(config, 1000).unwrap();

    assert!(index.is_empty());

    println!("[VERIFIED] HnswPurposeIndex::with_capacity pre-allocates");
}

// =========================================================================
// Insert Tests
// =========================================================================

#[test]
fn test_insert_single_entry() {
    let mut index = HnswPurposeIndex::new(purpose_config()).unwrap();
    let entry = create_entry(0.7, "master_ml");
    let memory_id = entry.memory_id;

    println!("[BEFORE] index.len()={}", index.len());

    index.insert(entry).unwrap();

    println!("[AFTER] index.len()={}", index.len());

    assert_eq!(index.len(), 1);
    assert!(index.contains(memory_id));

    println!("[VERIFIED] Single entry inserted correctly");
}

#[test]
fn test_insert_multiple_entries() {
    let mut index = HnswPurposeIndex::new(purpose_config()).unwrap();

    let entries: Vec<PurposeIndexEntry> = (0..10)
        .map(|i| create_entry(0.3 + i as f32 * 0.05, "goal"))
        .collect();

    for entry in &entries {
        index.insert(entry.clone()).unwrap();
    }

    assert_eq!(index.len(), 10);

    println!("[VERIFIED] Multiple entries inserted correctly");
}

#[test]
fn test_insert_updates_existing() {
    let mut index = HnswPurposeIndex::new(purpose_config()).unwrap();
    let memory_id = Uuid::new_v4();

    // First insert
    let entry1 = PurposeIndexEntry::new(
        memory_id,
        create_purpose_vector(0.5, 0.02),
        create_metadata("goal1"),
    );
    index.insert(entry1).unwrap();

    assert_eq!(index.len(), 1);
    let retrieved1 = index.get(memory_id).unwrap();
    assert_eq!(retrieved1.metadata.primary_goal.as_str(), "goal1");

    // Update with same ID, different goal
    let entry2 = PurposeIndexEntry::new(
        memory_id,
        create_purpose_vector(0.8, 0.01),
        create_metadata("goal2"),
    );
    index.insert(entry2).unwrap();

    assert_eq!(index.len(), 1); // Still only 1 entry
    let retrieved2 = index.get(memory_id).unwrap();
    assert_eq!(retrieved2.metadata.primary_goal.as_str(), "goal2");

    println!("[VERIFIED] Insert updates existing entry with same ID");
}

#[test]
fn test_insert_updates_secondary_indexes() {
    let mut index = HnswPurposeIndex::new(purpose_config()).unwrap();

    let entry1 = create_entry(0.5, "goal_a");
    let entry2 = create_entry(0.6, "goal_b");
    let entry3 = create_entry(0.7, "goal_a");

    index.insert(entry1.clone()).unwrap();
    index.insert(entry2.clone()).unwrap();
    index.insert(entry3.clone()).unwrap();

    // Check goal index
    assert_eq!(index.goal_count(), 2);
    let goal_a_set = index.get_by_goal(&GoalId::new("goal_a")).unwrap();
    assert_eq!(goal_a_set.len(), 2);
    assert!(goal_a_set.contains(&entry1.memory_id));
    assert!(goal_a_set.contains(&entry3.memory_id));

    println!("[VERIFIED] Insert updates secondary indexes correctly");
}

// =========================================================================
// Remove Tests
// =========================================================================

#[test]
fn test_remove_existing_entry() {
    let mut index = HnswPurposeIndex::new(purpose_config()).unwrap();
    let entry = create_entry(0.7, "test_goal");
    let memory_id = entry.memory_id;

    index.insert(entry).unwrap();
    assert_eq!(index.len(), 1);

    println!("[BEFORE REMOVE] index.len()={}", index.len());

    index.remove(memory_id).unwrap();

    println!("[AFTER REMOVE] index.len()={}", index.len());

    assert_eq!(index.len(), 0);
    assert!(!index.contains(memory_id));

    println!("[VERIFIED] Remove deletes entry correctly");
}

#[test]
fn test_remove_non_existent_fails() {
    let mut index = HnswPurposeIndex::new(purpose_config()).unwrap();
    let non_existent_id = Uuid::new_v4();

    let result = index.remove(non_existent_id);

    assert!(result.is_err());
    let err = result.unwrap_err();
    let msg = err.to_string();
    assert!(msg.contains(&non_existent_id.to_string()));

    println!(
        "[VERIFIED] FAIL FAST: Remove fails for non-existent entry: {}",
        msg
    );
}

#[test]
fn test_remove_updates_secondary_indexes() {
    let mut index = HnswPurposeIndex::new(purpose_config()).unwrap();

    let entry1 = create_entry(0.5, "shared_goal");
    let entry2 = create_entry(0.6, "shared_goal");
    let id1 = entry1.memory_id;

    index.insert(entry1).unwrap();
    index.insert(entry2).unwrap();

    assert_eq!(index.goal_count(), 1);
    assert_eq!(
        index
            .get_by_goal(&GoalId::new("shared_goal"))
            .unwrap()
            .len(),
        2
    );

    index.remove(id1).unwrap();

    assert_eq!(index.goal_count(), 1); // Goal still exists
    assert_eq!(
        index
            .get_by_goal(&GoalId::new("shared_goal"))
            .unwrap()
            .len(),
        1
    ); // But with one less member

    println!("[VERIFIED] Remove updates secondary indexes correctly");
}

#[test]
fn test_remove_cleans_up_empty_indexes() {
    let mut index = HnswPurposeIndex::new(purpose_config()).unwrap();

    let entry = create_entry(0.5, "unique_goal");
    let id = entry.memory_id;

    index.insert(entry).unwrap();
    assert_eq!(index.goal_count(), 1);

    index.remove(id).unwrap();

    // Empty sets should be removed
    assert_eq!(index.goal_count(), 0);
    assert!(index.get_by_goal(&GoalId::new("unique_goal")).is_none());

    println!("[VERIFIED] Remove cleans up empty secondary index entries");
}

// =========================================================================
// Get Tests
// =========================================================================

#[test]
fn test_get_existing_entry() {
    let mut index = HnswPurposeIndex::new(purpose_config()).unwrap();
    let entry = create_entry(0.75, "retrieve_goal");
    let memory_id = entry.memory_id;

    index.insert(entry.clone()).unwrap();

    let retrieved = index.get(memory_id).unwrap();

    assert_eq!(retrieved.memory_id, memory_id);
    assert_eq!(
        retrieved.metadata.primary_goal.as_str(),
        entry.metadata.primary_goal.as_str()
    );
    assert_eq!(
        retrieved.purpose_vector.alignments,
        entry.purpose_vector.alignments
    );

    println!("[VERIFIED] Get retrieves correct entry");
}

#[test]
fn test_get_non_existent_fails() {
    let index = HnswPurposeIndex::new(purpose_config()).unwrap();
    let non_existent_id = Uuid::new_v4();

    let result = index.get(non_existent_id);

    assert!(result.is_err());
    let err = result.unwrap_err();
    let msg = err.to_string();
    assert!(msg.contains("not found"));

    println!(
        "[VERIFIED] FAIL FAST: Get fails for non-existent entry: {}",
        msg
    );
}
