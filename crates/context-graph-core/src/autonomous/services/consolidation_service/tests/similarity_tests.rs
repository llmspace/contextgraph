//! Tests for compute_similarity and should_consolidate methods

use crate::autonomous::curation::{ConsolidationConfig, MemoryId};
use crate::autonomous::services::consolidation_service::{ConsolidationService, MemoryContent};

fn create_test_memory(embedding: Vec<f32>, text: &str, alignment: f32) -> MemoryContent {
    MemoryContent::new(MemoryId::new(), embedding, text.to_string(), alignment)
}

// ========== compute_similarity Tests ==========

#[test]
fn test_compute_similarity_identical() {
    let service = ConsolidationService::new();

    // Normalized vector
    let embedding = vec![0.6, 0.8]; // magnitude = 1.0
    let a = create_test_memory(embedding.clone(), "a", 0.8);
    let b = create_test_memory(embedding, "b", 0.8);

    let sim = service.compute_similarity(&a, &b);
    assert!(
        (sim - 1.0).abs() < 0.001,
        "Identical vectors should have similarity 1.0, got {}",
        sim
    );

    println!("[PASS] test_compute_similarity_identical");
}

#[test]
fn test_compute_similarity_orthogonal() {
    let service = ConsolidationService::new();

    let a = create_test_memory(vec![1.0, 0.0], "a", 0.8);
    let b = create_test_memory(vec![0.0, 1.0], "b", 0.8);

    let sim = service.compute_similarity(&a, &b);
    assert!(
        sim.abs() < 0.001,
        "Orthogonal vectors should have similarity 0.0, got {}",
        sim
    );

    println!("[PASS] test_compute_similarity_orthogonal");
}

#[test]
fn test_compute_similarity_real_calculation() {
    let service = ConsolidationService::new();

    // Two similar but not identical vectors
    let a = create_test_memory(vec![0.8, 0.6, 0.0], "a", 0.8);
    let b = create_test_memory(vec![0.7, 0.7, 0.1], "b", 0.8);

    // Manual calculation:
    // dot = 0.8*0.7 + 0.6*0.7 + 0.0*0.1 = 0.56 + 0.42 + 0.0 = 0.98
    // |a| = sqrt(0.64 + 0.36 + 0.0) = 1.0
    // |b| = sqrt(0.49 + 0.49 + 0.01) = sqrt(0.99) ≈ 0.995
    // sim = 0.98 / (1.0 * 0.995) ≈ 0.985

    let sim = service.compute_similarity(&a, &b);
    assert!(
        (sim - 0.985).abs() < 0.01,
        "Expected similarity ~0.985, got {}",
        sim
    );

    println!("[PASS] test_compute_similarity_real_calculation");
}

#[test]
fn test_compute_similarity_empty_embedding() {
    let service = ConsolidationService::new();

    let a = create_test_memory(vec![], "a", 0.8);
    let b = create_test_memory(vec![], "b", 0.8);

    let sim = service.compute_similarity(&a, &b);
    assert!(
        sim.abs() < f32::EPSILON,
        "Empty embeddings should return 0.0"
    );

    println!("[PASS] test_compute_similarity_empty_embedding");
}

#[test]
fn test_compute_similarity_different_dimensions() {
    let service = ConsolidationService::new();

    let a = create_test_memory(vec![1.0, 0.0, 0.0], "a", 0.8);
    let b = create_test_memory(vec![1.0, 0.0], "b", 0.8);

    let sim = service.compute_similarity(&a, &b);
    assert!(
        sim.abs() < f32::EPSILON,
        "Different dimensions should return 0.0"
    );

    println!("[PASS] test_compute_similarity_different_dimensions");
}

#[test]
fn test_compute_similarity_zero_magnitude() {
    let service = ConsolidationService::new();

    let a = create_test_memory(vec![0.0, 0.0, 0.0], "a", 0.8);
    let b = create_test_memory(vec![1.0, 0.0, 0.0], "b", 0.8);

    let sim = service.compute_similarity(&a, &b);
    assert!(sim.abs() < f32::EPSILON, "Zero magnitude should return 0.0");

    println!("[PASS] test_compute_similarity_zero_magnitude");
}

#[test]
fn test_compute_similarity_high_dimensional() {
    let service = ConsolidationService::new();

    // Simulate realistic 384-dim embedding
    let mut vec_a: Vec<f32> = (0..384).map(|i| (i as f32 * 0.01).sin()).collect();
    let mut vec_b: Vec<f32> = (0..384).map(|i| (i as f32 * 0.01 + 0.1).sin()).collect();

    // Normalize
    let mag_a: f32 = vec_a.iter().map(|x| x * x).sum::<f32>().sqrt();
    let mag_b: f32 = vec_b.iter().map(|x| x * x).sum::<f32>().sqrt();
    for v in &mut vec_a {
        *v /= mag_a;
    }
    for v in &mut vec_b {
        *v /= mag_b;
    }

    let a = create_test_memory(vec_a, "a", 0.8);
    let b = create_test_memory(vec_b, "b", 0.8);

    let sim = service.compute_similarity(&a, &b);
    assert!(
        sim > 0.9 && sim < 1.0,
        "Similar high-dim vectors should have high similarity: {}",
        sim
    );

    println!("[PASS] test_compute_similarity_high_dimensional");
}

// ========== should_consolidate Tests ==========

#[test]
fn test_should_consolidate_true() {
    let service = ConsolidationService::new();
    // Default: similarity >= 0.92, theta_diff <= 0.05

    assert!(service.should_consolidate(0.95, 0.03));
    assert!(service.should_consolidate(0.92, 0.05)); // Boundary

    println!("[PASS] test_should_consolidate_true");
}

#[test]
fn test_should_consolidate_false_low_similarity() {
    let service = ConsolidationService::new();

    assert!(!service.should_consolidate(0.90, 0.03));
    assert!(!service.should_consolidate(0.919, 0.03)); // Just below

    println!("[PASS] test_should_consolidate_false_low_similarity");
}

#[test]
fn test_should_consolidate_false_high_theta_diff() {
    let service = ConsolidationService::new();

    assert!(!service.should_consolidate(0.95, 0.10));
    assert!(!service.should_consolidate(0.95, 0.051)); // Just above

    println!("[PASS] test_should_consolidate_false_high_theta_diff");
}

#[test]
fn test_should_consolidate_custom_thresholds() {
    let config = ConsolidationConfig {
        enabled: true,
        similarity_threshold: 0.80,
        max_daily_merges: 50,
        theta_diff_threshold: 0.15,
    };
    let service = ConsolidationService::with_config(config);

    assert!(service.should_consolidate(0.85, 0.10));
    assert!(!service.should_consolidate(0.75, 0.10));
    assert!(!service.should_consolidate(0.85, 0.20));

    println!("[PASS] test_should_consolidate_custom_thresholds");
}
