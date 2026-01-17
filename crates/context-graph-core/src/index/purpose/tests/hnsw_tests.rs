//! HNSW Index Tests - Insert/remove/search cycle

use super::helpers::{
    create_clustered_entries, create_entry, create_purpose_vector, purpose_config,
};
use crate::index::purpose::entry::GoalId;
use crate::index::purpose::hnsw_purpose::{HnswPurposeIndex, PurposeIndexOps};
use crate::index::purpose::query::{PurposeQuery, PurposeQueryTarget};
use crate::types::JohariQuadrant;
use uuid::Uuid;

#[test]
fn test_hnsw_insert_remove_get_cycle() {
    let mut index = HnswPurposeIndex::new(purpose_config()).unwrap();
    let entry = create_entry(0.7, "test_goal", JohariQuadrant::Open);
    let memory_id = entry.memory_id;

    println!(
        "[BEFORE] len={}, contains({})={}",
        index.len(),
        memory_id,
        index.contains(memory_id)
    );

    // Insert
    index.insert(entry.clone()).unwrap();
    assert_eq!(index.len(), 1);
    assert!(index.contains(memory_id));

    println!(
        "[AFTER INSERT] len={}, contains({})={}",
        index.len(),
        memory_id,
        index.contains(memory_id)
    );

    // Get
    let retrieved = index.get(memory_id).unwrap();
    assert_eq!(retrieved.memory_id, memory_id);
    assert_eq!(retrieved.metadata.primary_goal.as_str(), "test_goal");

    // Remove
    index.remove(memory_id).unwrap();
    assert_eq!(index.len(), 0);
    assert!(!index.contains(memory_id));

    println!(
        "[AFTER REMOVE] len={}, contains({})={}",
        index.len(),
        memory_id,
        index.contains(memory_id)
    );

    println!("[VERIFIED] Insert/remove/get cycle works correctly");
}

#[test]
fn test_hnsw_search_with_vector_target() {
    let mut index = HnswPurposeIndex::new(purpose_config()).unwrap();

    // Insert entries with varying purpose vectors
    for i in 0..10 {
        let entry = create_entry(0.3 + i as f32 * 0.05, "goal", JohariQuadrant::Open);
        index.insert(entry).unwrap();
    }

    // Search with vector similar to highest entry
    let query_vector = create_purpose_vector(0.75, 0.02);
    let query = PurposeQuery::new(PurposeQueryTarget::Vector(query_vector), 5, 0.0).unwrap();

    let results = index.search(&query).unwrap();

    assert!(!results.is_empty());
    assert!(results.len() <= 5);

    // Results should be sorted by similarity descending
    for i in 1..results.len() {
        assert!(results[i - 1].purpose_similarity >= results[i].purpose_similarity);
    }

    println!("[VERIFIED] Search with Vector target returns sorted results");
}

#[test]
fn test_hnsw_search_with_pattern_target() {
    let mut index = HnswPurposeIndex::new(purpose_config()).unwrap();

    // Insert clustered entries
    let entries = create_clustered_entries();
    for entry in entries {
        index.insert(entry).unwrap();
    }

    let target = PurposeQueryTarget::pattern(2, 0.5).unwrap();
    let query = PurposeQuery::new(target, 20, 0.0).unwrap();

    let results = index.search(&query).unwrap();

    // Pattern search should return some results
    println!("[RESULT] Pattern search found {} results", results.len());

    println!("[VERIFIED] Search with Pattern target executes");
}

#[test]
fn test_hnsw_search_with_from_memory_target() {
    let mut index = HnswPurposeIndex::new(purpose_config()).unwrap();

    // Insert entries
    let entries: Vec<_> = (0..5)
        .map(|i| create_entry(0.4 + i as f32 * 0.1, "goal", JohariQuadrant::Open))
        .collect();

    for entry in &entries {
        index.insert(entry.clone()).unwrap();
    }

    // Search from existing memory
    let source_id = entries[2].memory_id;
    let query = PurposeQuery::new(PurposeQueryTarget::from_memory(source_id), 3, 0.0).unwrap();

    let results = index.search(&query).unwrap();

    assert!(!results.is_empty());
    // Should find the source itself
    assert!(results.iter().any(|r| r.memory_id == source_id));

    println!("[VERIFIED] Search with FromMemory target finds source memory");
}

#[test]
fn test_hnsw_search_with_goal_filtering() {
    let mut index = HnswPurposeIndex::new(purpose_config()).unwrap();

    // Insert entries with different goals
    for i in 0..10 {
        let goal = if i % 2 == 0 { "goal_a" } else { "goal_b" };
        let entry = create_entry(0.5 + i as f32 * 0.02, goal, JohariQuadrant::Open);
        index.insert(entry).unwrap();
    }

    let query = PurposeQuery::new(
        PurposeQueryTarget::Vector(create_purpose_vector(0.55, 0.02)),
        10,
        0.0,
    )
    .unwrap()
    .with_goal_filter(GoalId::new("goal_a"));

    let results = index.search(&query).unwrap();

    for result in &results {
        assert_eq!(result.metadata.primary_goal.as_str(), "goal_a");
    }

    println!(
        "[VERIFIED] Goal filtering returns only matching entries ({} results)",
        results.len()
    );
}

#[test]
fn test_hnsw_search_with_quadrant_filtering() {
    let mut index = HnswPurposeIndex::new(purpose_config()).unwrap();

    // Insert entries in different quadrants
    for quadrant in JohariQuadrant::all() {
        for i in 0..3 {
            let entry = create_entry(0.5 + i as f32 * 0.05, "goal", quadrant);
            index.insert(entry).unwrap();
        }
    }

    let query = PurposeQuery::new(
        PurposeQueryTarget::Vector(create_purpose_vector(0.55, 0.02)),
        10,
        0.0,
    )
    .unwrap()
    .with_quadrant_filter(JohariQuadrant::Hidden);

    let results = index.search(&query).unwrap();

    for result in &results {
        assert_eq!(result.metadata.dominant_quadrant, JohariQuadrant::Hidden);
    }

    println!(
        "[VERIFIED] Quadrant filtering returns only matching entries ({} results)",
        results.len()
    );
}

#[test]
fn test_hnsw_search_min_similarity_threshold() {
    let mut index = HnswPurposeIndex::new(purpose_config()).unwrap();

    for i in 0..10 {
        let entry = create_entry(0.1 + i as f32 * 0.08, "goal", JohariQuadrant::Open);
        index.insert(entry).unwrap();
    }

    let query = PurposeQuery::new(
        PurposeQueryTarget::Vector(create_purpose_vector(0.9, 0.01)),
        10,
        0.8, // High min_similarity
    )
    .unwrap();

    let results = index.search(&query).unwrap();

    for result in &results {
        assert!(
            result.purpose_similarity >= 0.8,
            "Similarity {} should be >= 0.8",
            result.purpose_similarity
        );
    }

    println!(
        "[VERIFIED] min_similarity threshold filters results ({} passed)",
        results.len()
    );
}

#[test]
fn test_hnsw_search_empty_index() {
    let index = HnswPurposeIndex::new(purpose_config()).unwrap();

    let query = PurposeQuery::new(
        PurposeQueryTarget::Vector(create_purpose_vector(0.5, 0.02)),
        10,
        0.0,
    )
    .unwrap();

    // Correct database semantics: searching an empty index returns empty results
    // Error should only occur on actual failures (network, disk, corruption, invalid input)
    let result = index.search(&query);
    assert!(
        result.is_ok(),
        "Search on empty index should succeed with empty results"
    );

    let results = result.unwrap();
    assert!(
        results.is_empty(),
        "Empty index should return empty results"
    );

    println!("[VERIFIED] Search on empty index returns empty results (correct database semantics)");
}

#[test]
fn test_hnsw_not_found_error_on_missing_memory() {
    let index = HnswPurposeIndex::new(purpose_config()).unwrap();
    let non_existent = Uuid::new_v4();

    // Get fails on missing
    let result = index.get(non_existent);
    assert!(result.is_err());
    let msg = result.unwrap_err().to_string();
    assert!(msg.contains("not found"));

    println!(
        "[VERIFIED] FAIL FAST: Get fails for non-existent memory: {}",
        msg
    );
}

#[test]
fn test_hnsw_remove_non_existent_fails() {
    let mut index = HnswPurposeIndex::new(purpose_config()).unwrap();
    let non_existent = Uuid::new_v4();

    let result = index.remove(non_existent);
    assert!(result.is_err());
    let msg = result.unwrap_err().to_string();
    assert!(msg.contains("not found"));

    println!(
        "[VERIFIED] FAIL FAST: Remove fails for non-existent memory: {}",
        msg
    );
}

#[test]
fn test_hnsw_duplicate_handling_updates_entry() {
    let mut index = HnswPurposeIndex::new(purpose_config()).unwrap();
    let memory_id = Uuid::new_v4();

    // First insert
    let entry1 = crate::index::purpose::entry::PurposeIndexEntry::new(
        memory_id,
        create_purpose_vector(0.5, 0.02),
        super::helpers::create_metadata("goal1", JohariQuadrant::Open),
    );
    index.insert(entry1).unwrap();
    assert_eq!(index.len(), 1);

    // Second insert with same ID (update)
    let entry2 = crate::index::purpose::entry::PurposeIndexEntry::new(
        memory_id,
        create_purpose_vector(0.8, 0.01),
        super::helpers::create_metadata("goal2", JohariQuadrant::Hidden),
    );
    index.insert(entry2).unwrap();

    // Should still have 1 entry, updated
    assert_eq!(index.len(), 1);
    let retrieved = index.get(memory_id).unwrap();
    assert_eq!(retrieved.metadata.primary_goal.as_str(), "goal2");
    assert_eq!(retrieved.metadata.dominant_quadrant, JohariQuadrant::Hidden);

    println!("[VERIFIED] Duplicate handling updates existing entry");
}

#[test]
fn test_hnsw_search_from_memory_non_existent_fails() {
    let index = HnswPurposeIndex::new(purpose_config()).unwrap();
    let non_existent = Uuid::new_v4();

    let query = PurposeQuery::new(PurposeQueryTarget::from_memory(non_existent), 10, 0.0).unwrap();
    let result = index.search(&query);

    assert!(result.is_err());
    let msg = result.unwrap_err().to_string();
    assert!(msg.contains("not found"));

    println!(
        "[VERIFIED] FAIL FAST: FromMemory search fails for non-existent memory: {}",
        msg
    );
}
