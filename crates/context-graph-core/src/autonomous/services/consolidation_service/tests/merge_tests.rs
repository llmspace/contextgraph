//! Tests for merge_memories and compute_combined_alignment methods

use crate::autonomous::curation::MemoryId;
use crate::autonomous::services::consolidation_service::{ConsolidationService, MemoryContent};

fn create_test_memory(embedding: Vec<f32>, text: &str, alignment: f32) -> MemoryContent {
    MemoryContent::new(MemoryId::new(), embedding, text.to_string(), alignment)
}

// ========== merge_memories Tests ==========

#[test]
fn test_merge_memories_empty() {
    let service = ConsolidationService::new();
    let merged = service.merge_memories(&[]);

    assert!(merged.embedding.is_empty());
    assert!(merged.text.is_empty());
    assert!(merged.alignment.abs() < f32::EPSILON);

    println!("[PASS] test_merge_memories_empty");
}

#[test]
fn test_merge_memories_single() {
    let service = ConsolidationService::new();

    let mem = create_test_memory(vec![1.0, 0.0], "original", 0.8).with_access_count(10);

    let merged = service.merge_memories(std::slice::from_ref(&mem));

    assert_eq!(merged.embedding, mem.embedding);
    assert_eq!(merged.text, "original");
    assert!((merged.alignment - 0.8).abs() < f32::EPSILON);

    println!("[PASS] test_merge_memories_single");
}

#[test]
fn test_merge_memories_text_combined() {
    let service = ConsolidationService::new();

    let a = create_test_memory(vec![1.0, 0.0], "first", 0.8);
    let b = create_test_memory(vec![1.0, 0.0], "second", 0.8);
    let c = create_test_memory(vec![1.0, 0.0], "third", 0.8);

    let merged = service.merge_memories(&[a, b, c]);

    assert!(merged.text.contains("first"));
    assert!(merged.text.contains("second"));
    assert!(merged.text.contains("third"));
    assert!(merged.text.contains(" | "));

    println!("[PASS] test_merge_memories_text_combined");
}

#[test]
fn test_merge_memories_embedding_averaged() {
    let service = ConsolidationService::new();

    // Two perpendicular unit vectors with equal weight
    let a = create_test_memory(vec![1.0, 0.0], "a", 0.8).with_access_count(0);
    let b = create_test_memory(vec![0.0, 1.0], "b", 0.8).with_access_count(0);

    let merged = service.merge_memories(&[a, b]);

    // Should be normalized average: [0.5, 0.5] normalized = [0.707, 0.707]
    assert_eq!(merged.dimension(), 2);
    let expected = 1.0 / (2.0_f32).sqrt();
    assert!(
        (merged.embedding[0] - expected).abs() < 0.01,
        "Expected {}, got {}",
        expected,
        merged.embedding[0]
    );
    assert!(
        (merged.embedding[1] - expected).abs() < 0.01,
        "Expected {}, got {}",
        expected,
        merged.embedding[1]
    );

    println!("[PASS] test_merge_memories_embedding_averaged");
}

#[test]
fn test_merge_memories_weighted_by_access() {
    let service = ConsolidationService::new();

    // Higher access count should have more influence
    let a = create_test_memory(vec![1.0, 0.0], "a", 0.8).with_access_count(9); // weight = 10
    let b = create_test_memory(vec![0.0, 1.0], "b", 0.8).with_access_count(0); // weight = 1

    let merged = service.merge_memories(&[a, b]);

    // a has 10x weight, so embedding should be closer to [1, 0]
    // Weighted: [10/11, 1/11] normalized
    assert!(
        merged.embedding[0] > merged.embedding[1],
        "First component should dominate: {:?}",
        merged.embedding
    );

    println!("[PASS] test_merge_memories_weighted_by_access");
}

#[test]
fn test_merge_memories_access_count_summed() {
    let service = ConsolidationService::new();

    let a = create_test_memory(vec![1.0], "a", 0.8).with_access_count(5);
    let b = create_test_memory(vec![1.0], "b", 0.8).with_access_count(10);
    let c = create_test_memory(vec![1.0], "c", 0.8).with_access_count(7);

    let merged = service.merge_memories(&[a, b, c]);
    assert_eq!(merged.access_count, 22);

    println!("[PASS] test_merge_memories_access_count_summed");
}

#[test]
fn test_merge_memories_empty_embeddings() {
    let service = ConsolidationService::new();

    let a = create_test_memory(vec![], "first", 0.8);
    let b = create_test_memory(vec![], "second", 0.7);

    let merged = service.merge_memories(&[a, b]);

    assert!(merged.embedding.is_empty());
    assert!(merged.text.contains("first"));
    assert!(merged.text.contains("second"));

    println!("[PASS] test_merge_memories_empty_embeddings");
}

// ========== compute_combined_alignment Tests ==========

#[test]
fn test_combined_alignment_empty() {
    let service = ConsolidationService::new();
    let result = service.compute_combined_alignment(&[]);
    assert!(result.abs() < f32::EPSILON);

    println!("[PASS] test_combined_alignment_empty");
}

#[test]
fn test_combined_alignment_single() {
    let service = ConsolidationService::new();
    let result = service.compute_combined_alignment(&[0.85]);
    assert!((result - 0.85).abs() < f32::EPSILON);

    println!("[PASS] test_combined_alignment_single");
}

#[test]
fn test_combined_alignment_equal() {
    let service = ConsolidationService::new();
    // Equal alignments should give that alignment
    let result = service.compute_combined_alignment(&[0.80, 0.80, 0.80]);
    assert!((result - 0.80).abs() < f32::EPSILON);

    println!("[PASS] test_combined_alignment_equal");
}

#[test]
fn test_combined_alignment_favors_higher() {
    let service = ConsolidationService::new();

    // With [0.9, 0.5], weights are [0.81, 0.25]
    // weighted sum = 0.9*0.81 + 0.5*0.25 = 0.729 + 0.125 = 0.854
    // total weight = 1.06
    // result = 0.854 / 1.06 ≈ 0.806
    let result = service.compute_combined_alignment(&[0.9, 0.5]);

    // Result should be closer to 0.9 than 0.5
    assert!(
        result > 0.7,
        "Result should favor higher alignment: {}",
        result
    );
    assert!(result < 0.9, "Result should not exceed highest: {}", result);

    println!("[PASS] test_combined_alignment_favors_higher");
}

#[test]
fn test_combined_alignment_zero_weights() {
    let service = ConsolidationService::new();
    // All zeros should use simple average = 0.0
    let result = service.compute_combined_alignment(&[0.0, 0.0]);
    assert!(result.abs() < f32::EPSILON);

    println!("[PASS] test_combined_alignment_zero_weights");
}

#[test]
fn test_combined_alignment_real_calculation() {
    let service = ConsolidationService::new();

    // Manual calculation: [0.8, 0.7, 0.6]
    // weights = [0.64, 0.49, 0.36] sum = 1.49
    // weighted = 0.8*0.64 + 0.7*0.49 + 0.6*0.36 = 0.512 + 0.343 + 0.216 = 1.071
    // result = 1.071 / 1.49 ≈ 0.7188

    let result = service.compute_combined_alignment(&[0.8, 0.7, 0.6]);
    assert!(
        (result - 0.7188).abs() < 0.01,
        "Expected ~0.7188, got {}",
        result
    );

    println!("[PASS] test_combined_alignment_real_calculation");
}
