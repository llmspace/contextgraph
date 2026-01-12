//! Tests for type definitions (MemoryContent, MemoryPair, ServiceConsolidationCandidate)

use crate::autonomous::curation::MemoryId;
use crate::autonomous::services::consolidation_service::{
    MemoryContent, MemoryPair, ServiceConsolidationCandidate,
};

fn create_test_memory(embedding: Vec<f32>, text: &str, alignment: f32) -> MemoryContent {
    MemoryContent::new(MemoryId::new(), embedding, text.to_string(), alignment)
}

// ========== MemoryContent Tests ==========

#[test]
fn test_memory_content_new() {
    let embedding = vec![0.5, 0.5, 0.5];
    let mem = create_test_memory(embedding.clone(), "test content", 0.8);

    assert_eq!(mem.embedding, embedding);
    assert_eq!(mem.text, "test content");
    assert!((mem.alignment - 0.8).abs() < f32::EPSILON);
    assert_eq!(mem.access_count, 0);

    println!("[PASS] test_memory_content_new");
}

#[test]
fn test_memory_content_with_access_count() {
    let mem = create_test_memory(vec![1.0, 0.0], "test", 0.5).with_access_count(42);

    assert_eq!(mem.access_count, 42);

    println!("[PASS] test_memory_content_with_access_count");
}

#[test]
fn test_memory_content_dimension() {
    let mem = create_test_memory(vec![1.0, 2.0, 3.0, 4.0], "test", 0.7);
    assert_eq!(mem.dimension(), 4);

    let empty_mem = create_test_memory(vec![], "empty", 0.5);
    assert_eq!(empty_mem.dimension(), 0);

    println!("[PASS] test_memory_content_dimension");
}

// ========== MemoryPair Tests ==========

#[test]
fn test_memory_pair_new() {
    let a = create_test_memory(vec![1.0], "a", 0.8);
    let b = create_test_memory(vec![1.0], "b", 0.6);

    let pair = MemoryPair::new(a.clone(), b.clone());
    assert_eq!(pair.a.text, "a");
    assert_eq!(pair.b.text, "b");

    println!("[PASS] test_memory_pair_new");
}

#[test]
fn test_memory_pair_alignment_diff() {
    let a = create_test_memory(vec![1.0], "a", 0.9);
    let b = create_test_memory(vec![1.0], "b", 0.7);

    let pair = MemoryPair::new(a, b);
    assert!((pair.alignment_diff() - 0.2).abs() < f32::EPSILON);

    // Test symmetry
    let c = create_test_memory(vec![1.0], "c", 0.3);
    let d = create_test_memory(vec![1.0], "d", 0.8);
    let pair2 = MemoryPair::new(c, d);
    assert!((pair2.alignment_diff() - 0.5).abs() < f32::EPSILON);

    println!("[PASS] test_memory_pair_alignment_diff");
}

// ========== ServiceConsolidationCandidate Tests ==========

#[test]
fn test_consolidation_candidate_new() {
    let id1 = MemoryId::new();
    let id2 = MemoryId::new();
    let target = MemoryId::new();

    let candidate = ServiceConsolidationCandidate::new(
        vec![id1.clone(), id2.clone()],
        target.clone(),
        0.95,
        0.85,
    );

    assert_eq!(candidate.source_ids.len(), 2);
    assert!((candidate.similarity - 0.95).abs() < f32::EPSILON);
    assert!((candidate.combined_alignment - 0.85).abs() < f32::EPSILON);

    println!("[PASS] test_consolidation_candidate_new");
}
