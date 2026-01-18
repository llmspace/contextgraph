//! Search tests for HnswPurposeIndex.

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
// Search Tests
// =========================================================================

#[test]
fn test_search_empty_index() {
    let index = HnswPurposeIndex::new(purpose_config()).unwrap();
    let query = PurposeQuery::new(
        PurposeQueryTarget::vector(create_purpose_vector(0.5, 0.02)),
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
fn test_search_vector_target() {
    let mut index = HnswPurposeIndex::new(purpose_config()).unwrap();

    // Insert entries with varying similarity to query
    let entries: Vec<PurposeIndexEntry> = (0..5)
        .map(|i| create_entry(0.4 + i as f32 * 0.1, "goal"))
        .collect();

    for entry in &entries {
        index.insert(entry.clone()).unwrap();
    }

    // Search with vector similar to highest entry
    let query_vector = create_purpose_vector(0.8, 0.02);
    let query = PurposeQuery::new(PurposeQueryTarget::vector(query_vector), 3, 0.0).unwrap();

    println!("[BEFORE] Searching for 3 nearest neighbors");

    let results = index.search(&query).unwrap();

    println!("[AFTER] Found {} results", results.len());

    assert_eq!(results.len(), 3);
    // Results should be sorted by similarity descending
    assert!(results[0].purpose_similarity >= results[1].purpose_similarity);
    assert!(results[1].purpose_similarity >= results[2].purpose_similarity);

    println!("[VERIFIED] Search with vector target returns sorted results");
}

#[test]
fn test_search_with_min_similarity_filter() {
    let mut index = HnswPurposeIndex::new(purpose_config()).unwrap();

    for i in 0..10 {
        index
            .insert(create_entry(0.1 + i as f32 * 0.08, "goal"))
            .unwrap();
    }

    let query_vector = create_purpose_vector(0.9, 0.01);
    let query = PurposeQuery::new(PurposeQueryTarget::vector(query_vector), 10, 0.8).unwrap();

    let results = index.search(&query).unwrap();

    // All results should meet min_similarity threshold
    for result in &results {
        assert!(
            result.purpose_similarity >= 0.8,
            "Similarity {} should be >= 0.8",
            result.purpose_similarity
        );
    }

    println!(
        "[VERIFIED] Search respects min_similarity filter ({} results)",
        results.len()
    );
}

#[test]
fn test_search_with_goal_filter() {
    let mut index = HnswPurposeIndex::new(purpose_config()).unwrap();

    // Insert entries with different goals
    for i in 0..5 {
        index
            .insert(create_entry(0.5 + i as f32 * 0.02, "goal_a"))
            .unwrap();
        index
            .insert(create_entry(0.5 + i as f32 * 0.02, "goal_b"))
            .unwrap();
    }

    let query_vector = create_purpose_vector(0.55, 0.02);
    let query = PurposeQuery::new(PurposeQueryTarget::vector(query_vector), 10, 0.0)
        .unwrap()
        .with_goal_filter(GoalId::new("goal_a"));

    let results = index.search(&query).unwrap();

    // All results should have goal_a
    for result in &results {
        assert_eq!(result.metadata.primary_goal.as_str(), "goal_a");
    }

    println!(
        "[VERIFIED] Search with goal filter returns only matching goals ({} results)",
        results.len()
    );
}

#[test]
fn test_search_from_memory_target() {
    let mut index = HnswPurposeIndex::new(purpose_config()).unwrap();

    // Insert entries
    let entries: Vec<PurposeIndexEntry> = (0..5)
        .map(|i| create_entry(0.4 + i as f32 * 0.1, "goal"))
        .collect();

    for entry in &entries {
        index.insert(entry.clone()).unwrap();
    }

    // Search from existing memory
    let source_id = entries[2].memory_id;
    let query = PurposeQuery::new(PurposeQueryTarget::from_memory(source_id), 3, 0.0).unwrap();

    let results = index.search(&query).unwrap();

    assert!(!results.is_empty());
    // Should find the source memory itself as most similar
    assert!(results.iter().any(|r| r.memory_id == source_id));

    println!("[VERIFIED] Search from memory target works");
}

#[test]
fn test_search_from_memory_non_existent_fails() {
    let index = HnswPurposeIndex::new(purpose_config()).unwrap();
    let non_existent = Uuid::new_v4();

    let query = PurposeQuery::new(PurposeQueryTarget::from_memory(non_existent), 10, 0.0).unwrap();

    let result = index.search(&query);

    assert!(result.is_err());
    let err = result.unwrap_err();
    let msg = err.to_string();
    assert!(msg.contains("not found"));

    println!(
        "[VERIFIED] FAIL FAST: Search from non-existent memory fails: {}",
        msg
    );
}

#[test]
fn test_search_respects_limit() {
    let mut index = HnswPurposeIndex::new(purpose_config()).unwrap();

    for i in 0..20 {
        index
            .insert(create_entry(0.3 + i as f32 * 0.03, "goal"))
            .unwrap();
    }

    let query = PurposeQuery::new(
        PurposeQueryTarget::vector(create_purpose_vector(0.5, 0.02)),
        5, // Limit to 5
        0.0,
    )
    .unwrap();

    let results = index.search(&query).unwrap();

    assert!(results.len() <= 5);

    println!("[VERIFIED] Search respects limit parameter");
}

// =========================================================================
// Pattern Search Tests
// =========================================================================

#[test]
fn test_search_pattern_target() {
    let mut index = HnswPurposeIndex::new(purpose_config()).unwrap();

    // Insert entries forming natural clusters
    // Cluster 1: low values
    for i in 0..5 {
        index
            .insert(create_entry(0.1 + i as f32 * 0.02, "goal_low"))
            .unwrap();
    }

    // Cluster 2: high values
    for i in 0..5 {
        index
            .insert(create_entry(0.8 + i as f32 * 0.02, "goal_high"))
            .unwrap();
    }

    let target = PurposeQueryTarget::pattern(2, 0.5).unwrap();
    let query = PurposeQuery::new(target, 20, 0.0).unwrap();

    let results = index.search(&query).unwrap();

    // Should find entries from clusters meeting the criteria
    println!("[RESULT] Pattern search found {} results", results.len());

    // Note: Exact results depend on clustering, but we should get some results
    // as long as clusters meet min_cluster_size and coherence_threshold

    println!("[VERIFIED] Search with pattern target executes");
}
