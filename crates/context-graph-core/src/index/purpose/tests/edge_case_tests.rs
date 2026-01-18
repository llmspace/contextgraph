//! Additional Edge Case Tests

use super::helpers::{create_entry, create_purpose_vector, purpose_config};
use crate::index::config::DistanceMetric;
use crate::index::config::HnswConfig;
use crate::index::purpose::entry::GoalId;
use crate::index::purpose::hnsw_purpose::{HnswPurposeIndex, PurposeIndexOps};
use crate::index::purpose::query::{PurposeQuery, PurposeQueryTarget};
use std::collections::HashSet;

#[test]
fn test_index_with_wrong_dimension_config_fails() {
    let wrong_config = HnswConfig::new(16, 200, 100, DistanceMetric::Cosine, 100);
    let result = HnswPurposeIndex::new(wrong_config);

    assert!(result.is_err());
    let msg = result.unwrap_err().to_string();
    assert!(msg.contains("13"));
    assert!(msg.contains("100"));

    println!(
        "[VERIFIED] FAIL FAST: Index rejects wrong dimension config: {}",
        msg
    );
}

#[test]
fn test_search_results_sorted_by_similarity_descending() {
    let mut index = HnswPurposeIndex::new(purpose_config()).unwrap();

    for i in 0..10 {
        let entry = create_entry(0.1 * i as f32, "goal");
        index.insert(entry).unwrap();
    }

    let query = PurposeQuery::new(
        PurposeQueryTarget::Vector(create_purpose_vector(0.5, 0.01)),
        10,
        0.0,
    )
    .unwrap();

    let results = index.search(&query).unwrap();

    for i in 1..results.len() {
        assert!(
            results[i - 1].purpose_similarity >= results[i].purpose_similarity,
            "Results should be sorted by similarity descending"
        );
    }

    println!("[VERIFIED] Search results sorted by similarity descending");
}

#[test]
fn test_filter_returns_empty_when_no_matches() {
    let mut index = HnswPurposeIndex::new(purpose_config()).unwrap();

    let entry = create_entry(0.5, "existing_goal");
    index.insert(entry).unwrap();

    let query = PurposeQuery::new(
        PurposeQueryTarget::Vector(create_purpose_vector(0.5, 0.02)),
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
fn test_search_respects_limit() {
    let mut index = HnswPurposeIndex::new(purpose_config()).unwrap();

    for i in 0..20 {
        let entry = create_entry(0.3 + i as f32 * 0.03, "goal");
        index.insert(entry).unwrap();
    }

    let query = PurposeQuery::new(
        PurposeQueryTarget::Vector(create_purpose_vector(0.5, 0.02)),
        5, // Limit to 5
        0.0,
    )
    .unwrap();

    let results = index.search(&query).unwrap();

    assert!(results.len() <= 5);

    println!(
        "[VERIFIED] Search respects limit parameter ({} results)",
        results.len()
    );
}

#[test]
fn test_goals_list_returns_all_goals() {
    let mut index = HnswPurposeIndex::new(purpose_config()).unwrap();

    index.insert(create_entry(0.5, "alpha")).unwrap();
    index.insert(create_entry(0.6, "beta")).unwrap();
    index.insert(create_entry(0.7, "gamma")).unwrap();

    let goals = index.goals();

    assert_eq!(goals.len(), 3);
    let goal_strs: HashSet<&str> = goals.iter().map(|g| g.as_str()).collect();
    assert!(goal_strs.contains("alpha"));
    assert!(goal_strs.contains("beta"));
    assert!(goal_strs.contains("gamma"));

    println!("[VERIFIED] goals() returns all distinct goals");
}

#[test]
fn test_multiple_inserts_and_removes() {
    let mut index = HnswPurposeIndex::new(purpose_config()).unwrap();

    // Insert 20 entries
    let mut ids: Vec<uuid::Uuid> = Vec::new();
    for i in 0..20 {
        let entry = create_entry(0.3 + i as f32 * 0.03, "goal");
        ids.push(entry.memory_id);
        index.insert(entry).unwrap();
    }
    assert_eq!(index.len(), 20);

    // Remove every other entry
    for i in (0..20).step_by(2) {
        index.remove(ids[i]).unwrap();
    }
    assert_eq!(index.len(), 10);

    // Verify remaining entries exist
    for i in (1..20).step_by(2) {
        assert!(index.contains(ids[i]));
    }

    // Verify removed entries don't exist
    for i in (0..20).step_by(2) {
        assert!(!index.contains(ids[i]));
    }

    println!("[VERIFIED] Multiple inserts and removes maintain consistency");
}
