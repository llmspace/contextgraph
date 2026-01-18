//! Full State Verification Tests - Complete workflow

use super::helpers::{create_entry_with_id, create_purpose_vector, purpose_config};
use crate::index::purpose::entry::{GoalId, PurposeIndexEntry, PurposeMetadata};
use crate::index::purpose::hnsw_purpose::{HnswPurposeIndex, PurposeIndexOps};
use crate::index::purpose::query::{PurposeQuery, PurposeQueryTarget, PurposeSearchResult};
use crate::types::fingerprint::PurposeVector;
use std::time::{Duration, SystemTime};
use uuid::Uuid;

#[test]
fn test_full_state_verification_complete_workflow() {
    let mut index = HnswPurposeIndex::new(purpose_config()).unwrap();

    // Initial state
    assert!(index.is_empty());
    assert_eq!(index.len(), 0);
    assert_eq!(index.goal_count(), 0);
    println!(
        "[STATE] Initial: empty={}, len={}, goals={}",
        index.is_empty(),
        index.len(),
        index.goal_count()
    );

    // Step 1: Insert multiple entries with diverse goals
    let entries: Vec<PurposeIndexEntry> = vec![
        create_entry_with_id(Uuid::new_v4(), 0.3, "goal_alpha"),
        create_entry_with_id(Uuid::new_v4(), 0.4, "goal_alpha"),
        create_entry_with_id(Uuid::new_v4(), 0.5, "goal_beta"),
        create_entry_with_id(Uuid::new_v4(), 0.6, "goal_beta"),
        create_entry_with_id(Uuid::new_v4(), 0.7, "goal_gamma"),
    ];

    let ids: Vec<Uuid> = entries.iter().map(|e| e.memory_id).collect();

    for entry in &entries {
        index.insert(entry.clone()).unwrap();
    }

    // Verify after inserts
    assert_eq!(index.len(), 5);
    assert_eq!(index.goal_count(), 3); // alpha, beta, gamma
    for id in &ids {
        assert!(index.contains(*id));
    }
    println!(
        "[STATE] After 5 inserts: len={}, goals={}",
        index.len(),
        index.goal_count()
    );

    // Step 2: Verify secondary indexes updated
    let alpha_set = index.get_by_goal(&GoalId::new("goal_alpha")).unwrap();
    assert_eq!(alpha_set.len(), 2);
    println!("[STATE] Secondary indexes: goal_alpha={}", alpha_set.len());

    // Step 3: Search with each query type
    // Vector search
    let vector_query = PurposeQuery::new(
        PurposeQueryTarget::Vector(create_purpose_vector(0.5, 0.02)),
        5,
        0.0,
    )
    .unwrap();
    let vector_results = index.search(&vector_query).unwrap();
    assert_eq!(vector_results.len(), 5);
    println!("[STATE] Vector search: {} results", vector_results.len());

    // FromMemory search
    let from_memory_query =
        PurposeQuery::new(PurposeQueryTarget::from_memory(ids[2]), 3, 0.0).unwrap();
    let from_memory_results = index.search(&from_memory_query).unwrap();
    assert!(!from_memory_results.is_empty());
    println!(
        "[STATE] FromMemory search: {} results",
        from_memory_results.len()
    );

    // Filtered search
    let filtered_query = PurposeQuery::new(
        PurposeQueryTarget::Vector(create_purpose_vector(0.5, 0.02)),
        10,
        0.0,
    )
    .unwrap()
    .with_goal_filter(GoalId::new("goal_beta"));
    let filtered_results = index.search(&filtered_query).unwrap();
    assert_eq!(filtered_results.len(), 2);
    println!(
        "[STATE] Filtered search (goal_beta): {} results",
        filtered_results.len()
    );

    // Step 4: Remove entries and verify cleanup
    let to_remove = ids[0]; // goal_alpha
    index.remove(to_remove).unwrap();

    assert_eq!(index.len(), 4);
    assert!(!index.contains(to_remove));
    let alpha_set = index.get_by_goal(&GoalId::new("goal_alpha")).unwrap();
    assert_eq!(alpha_set.len(), 1); // Reduced from 2 to 1
    println!(
        "[STATE] After remove: len={}, goal_alpha={}",
        index.len(),
        alpha_set.len()
    );

    // Step 5: Clear and verify
    index.clear();
    assert!(index.is_empty());
    assert_eq!(index.len(), 0);
    assert_eq!(index.goal_count(), 0);
    println!(
        "[STATE] After clear: empty={}, len={}, goals={}",
        index.is_empty(),
        index.len(),
        index.goal_count()
    );

    println!("[VERIFIED] Full state verification complete - all data structures consistent");
}

#[test]
fn test_full_state_secondary_indexes_consistency() {
    let mut index = HnswPurposeIndex::new(purpose_config()).unwrap();

    // Insert entries
    let entries = vec![
        create_entry_with_id(Uuid::new_v4(), 0.3, "shared_goal"),
        create_entry_with_id(Uuid::new_v4(), 0.5, "shared_goal"),
        create_entry_with_id(Uuid::new_v4(), 0.7, "unique_goal"),
    ];

    let ids: Vec<Uuid> = entries.iter().map(|e| e.memory_id).collect();
    for entry in entries {
        index.insert(entry).unwrap();
    }

    // Verify initial state
    assert_eq!(index.goal_count(), 2);
    assert!(index.get_by_goal(&GoalId::new("shared_goal")).is_some());
    assert!(index.get_by_goal(&GoalId::new("unique_goal")).is_some());

    // Remove one shared_goal entry
    index.remove(ids[0]).unwrap();
    let shared_set = index.get_by_goal(&GoalId::new("shared_goal")).unwrap();
    assert_eq!(shared_set.len(), 1); // Still exists with 1 member

    // Remove the unique_goal entry
    index.remove(ids[2]).unwrap();
    assert!(index.get_by_goal(&GoalId::new("unique_goal")).is_none()); // Empty set removed
    assert_eq!(index.goal_count(), 1);

    // Remove the last shared_goal entry
    index.remove(ids[1]).unwrap();
    assert!(index.get_by_goal(&GoalId::new("shared_goal")).is_none()); // Empty set removed
    assert_eq!(index.goal_count(), 0);

    println!("[VERIFIED] Secondary indexes cleaned up correctly on removal");
}

#[test]
fn test_full_state_results_contain_complete_data() {
    let mut index = HnswPurposeIndex::new(purpose_config()).unwrap();

    let alignments = [
        0.8, 0.7, 0.9, 0.6, 0.75, 0.65, 0.85, 0.72, 0.78, 0.68, 0.82, 0.71, 0.76,
    ];
    let pv = PurposeVector::new(alignments);
    let metadata = PurposeMetadata::new(GoalId::new("complete_test"), 0.95).unwrap();
    let entry = PurposeIndexEntry::new(Uuid::new_v4(), pv.clone(), metadata);
    let memory_id = entry.memory_id;

    index.insert(entry).unwrap();

    let query = PurposeQuery::new(PurposeQueryTarget::Vector(pv.clone()), 1, 0.0).unwrap();

    let results = index.search(&query).unwrap();
    assert_eq!(results.len(), 1);

    let result = &results[0];
    assert_eq!(result.memory_id, memory_id);
    assert_eq!(result.purpose_vector.alignments, alignments);
    assert_eq!(result.metadata.primary_goal.as_str(), "complete_test");
    assert!((result.metadata.confidence - 0.95).abs() < f32::EPSILON);

    println!("[VERIFIED] Search results contain complete entry data");
}

#[test]
fn test_purpose_search_result_matches_methods() {
    let pv = create_purpose_vector(0.8, 0.02);
    let metadata = super::helpers::create_metadata("test_goal");
    let result = PurposeSearchResult::new(Uuid::new_v4(), 0.95, pv, metadata);

    assert!(result.matches_goal(&GoalId::new("test_goal")));
    assert!(!result.matches_goal(&GoalId::new("other_goal")));

    println!("[VERIFIED] PurposeSearchResult matches_goal method works correctly");
}

#[test]
fn test_entry_stale_detection() {
    let past = SystemTime::now() - Duration::from_secs(3600);
    let metadata = PurposeMetadata::with_timestamp(GoalId::new("test"), 0.75, past).unwrap();

    let entry = PurposeIndexEntry::new(Uuid::new_v4(), PurposeVector::default(), metadata);

    // Entry is 1 hour old
    assert!(entry.is_stale(Duration::from_secs(1800))); // 30 min threshold
    assert!(!entry.is_stale(Duration::from_secs(7200))); // 2 hour threshold

    println!("[VERIFIED] Entry stale detection works correctly");
}
