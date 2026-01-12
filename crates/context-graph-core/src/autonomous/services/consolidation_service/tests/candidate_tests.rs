//! Tests for find_consolidation_candidates method

use crate::autonomous::curation::{ConsolidationConfig, MemoryId};
use crate::autonomous::services::consolidation_service::{
    ConsolidationService, MemoryContent, MemoryPair,
};

fn create_test_memory(embedding: Vec<f32>, text: &str, alignment: f32) -> MemoryContent {
    MemoryContent::new(MemoryId::new(), embedding, text.to_string(), alignment)
}

#[test]
fn test_find_candidates_empty() {
    let service = ConsolidationService::new();
    let candidates = service.find_consolidation_candidates(&[]);
    assert!(candidates.is_empty());

    println!("[PASS] test_find_candidates_empty");
}

#[test]
fn test_find_candidates_disabled() {
    let config = ConsolidationConfig {
        enabled: false,
        ..Default::default()
    };
    let service = ConsolidationService::with_config(config);

    // Even with identical memories, should return empty
    let embedding = vec![1.0, 0.0, 0.0];
    let a = create_test_memory(embedding.clone(), "a", 0.8);
    let b = create_test_memory(embedding, "b", 0.8);

    let pairs = vec![MemoryPair::new(a, b)];
    let candidates = service.find_consolidation_candidates(&pairs);
    assert!(candidates.is_empty());

    println!("[PASS] test_find_candidates_disabled");
}

#[test]
fn test_find_candidates_found() {
    let service = ConsolidationService::new();

    // Create nearly identical memories
    let embedding = vec![0.6, 0.8, 0.0];
    let a = create_test_memory(embedding.clone(), "a", 0.82);
    let b = create_test_memory(embedding, "b", 0.80); // diff = 0.02 < 0.05

    let pairs = vec![MemoryPair::new(a, b)];
    let candidates = service.find_consolidation_candidates(&pairs);

    assert_eq!(candidates.len(), 1);
    assert!((candidates[0].similarity - 1.0).abs() < 0.001);
    assert_eq!(candidates[0].source_ids.len(), 2);

    println!("[PASS] test_find_candidates_found");
}

#[test]
fn test_find_candidates_filtered() {
    let service = ConsolidationService::new();

    // Pair 1: High similarity, low diff -> should be found
    let a1 = create_test_memory(vec![1.0, 0.0], "a1", 0.80);
    let b1 = create_test_memory(vec![1.0, 0.0], "b1", 0.78);

    // Pair 2: Low similarity -> should not be found
    let a2 = create_test_memory(vec![1.0, 0.0], "a2", 0.80);
    let b2 = create_test_memory(vec![0.0, 1.0], "b2", 0.80);

    // Pair 3: High similarity but high diff -> should not be found
    let a3 = create_test_memory(vec![1.0, 0.0], "a3", 0.90);
    let b3 = create_test_memory(vec![1.0, 0.0], "b3", 0.70);

    let pairs = vec![
        MemoryPair::new(a1, b1),
        MemoryPair::new(a2, b2),
        MemoryPair::new(a3, b3),
    ];

    let candidates = service.find_consolidation_candidates(&pairs);
    assert_eq!(candidates.len(), 1);

    println!("[PASS] test_find_candidates_filtered");
}
