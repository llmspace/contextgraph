//! Integration tests for the consolidation service

use crate::autonomous::curation::{ConsolidationConfig, MemoryId};
use crate::autonomous::services::consolidation_service::{
    ConsolidationService, MemoryContent, MemoryPair,
};

fn create_test_memory(embedding: Vec<f32>, text: &str, alignment: f32) -> MemoryContent {
    MemoryContent::new(MemoryId::new(), embedding, text.to_string(), alignment)
}

#[test]
fn test_full_consolidation_workflow() {
    let mut service = ConsolidationService::new();

    // Create similar memories
    let embedding = vec![0.6, 0.8, 0.0];
    let mem1 =
        create_test_memory(embedding.clone(), "The quick brown fox", 0.85).with_access_count(5);
    let mem2 =
        create_test_memory(embedding.clone(), "The fast brown fox", 0.83).with_access_count(3);

    // Create dissimilar memory
    let mem3 = create_test_memory(vec![0.0, 0.0, 1.0], "Something unrelated", 0.90)
        .with_access_count(10);

    // Build pairs
    let pairs = vec![
        MemoryPair::new(mem1.clone(), mem2.clone()),
        MemoryPair::new(mem1.clone(), mem3.clone()),
    ];

    // Find candidates
    let candidates = service.find_consolidation_candidates(&pairs);
    assert_eq!(candidates.len(), 1, "Should find exactly one similar pair");

    // Verify candidate properties
    let candidate = &candidates[0];
    assert!(candidate.similarity > 0.99);
    assert!(candidate.combined_alignment > 0.83);

    // Consolidate
    let report = service.consolidate(&candidates);
    assert_eq!(report.merged, 1);
    assert_eq!(report.skipped, 0);

    // Verify merge result
    let merged = service.merge_memories(&[mem1.clone(), mem2.clone()]);
    assert!(merged.text.contains("quick"));
    assert!(merged.text.contains("fast"));
    assert_eq!(merged.access_count, 8);

    println!("[PASS] test_full_consolidation_workflow");
}

#[test]
fn test_batch_consolidation() {
    let config = ConsolidationConfig {
        enabled: true,
        similarity_threshold: 0.90,
        max_daily_merges: 100,
        theta_diff_threshold: 0.10,
    };
    let mut service = ConsolidationService::with_config(config);

    // Create multiple similar pairs
    let mut pairs = Vec::new();
    for i in 0..10 {
        let embedding = vec![1.0, 0.0, 0.0];
        let a = create_test_memory(embedding.clone(), &format!("Memory A{}", i), 0.80);
        let b = create_test_memory(embedding, &format!("Memory B{}", i), 0.78);
        pairs.push(MemoryPair::new(a, b));
    }

    let candidates = service.find_consolidation_candidates(&pairs);
    assert_eq!(candidates.len(), 10);

    let report = service.consolidate(&candidates);
    assert_eq!(report.merged, 10);
    assert!(!report.daily_limit_reached);

    println!("[PASS] test_batch_consolidation");
}

#[test]
fn test_edge_case_all_same_alignment() {
    let service = ConsolidationService::new();

    let embedding = vec![1.0, 0.0];
    let mems: Vec<MemoryContent> = (0..5)
        .map(|i| create_test_memory(embedding.clone(), &format!("mem{}", i), 0.75))
        .collect();

    let result = service
        .compute_combined_alignment(&mems.iter().map(|m| m.alignment).collect::<Vec<_>>());
    assert!((result - 0.75).abs() < f32::EPSILON);

    println!("[PASS] test_edge_case_all_same_alignment");
}
