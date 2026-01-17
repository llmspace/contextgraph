//! Cosine similarity tests.

use crate::alignment::calculator::similarity;
use crate::alignment::calculator::DefaultAlignmentCalculator;
use crate::purpose::GoalLevel;

#[test]
fn test_cosine_similarity_identical() {
    let a = vec![1.0, 0.0, 0.0];
    let b = vec![1.0, 0.0, 0.0];
    let sim = similarity::cosine_similarity(&a, &b);
    assert!((sim - 1.0).abs() < 0.001);
    println!("[VERIFIED] cosine_similarity: identical vectors = 1.0");
}

#[test]
fn test_cosine_similarity_orthogonal() {
    let a = vec![1.0, 0.0, 0.0];
    let b = vec![0.0, 1.0, 0.0];
    let sim = similarity::cosine_similarity(&a, &b);
    assert!(sim.abs() < 0.001);
    println!("[VERIFIED] cosine_similarity: orthogonal vectors = 0.0");
}

#[test]
fn test_cosine_similarity_opposite() {
    let a = vec![1.0, 0.0];
    let b = vec![-1.0, 0.0];
    let sim = similarity::cosine_similarity(&a, &b);
    assert!((sim - (-1.0)).abs() < 0.001);
    println!("[VERIFIED] cosine_similarity: opposite vectors = -1.0");
}

#[test]
fn test_cosine_similarity_mismatched_dims() {
    let a = vec![1.0, 0.0];
    let b = vec![1.0, 0.0, 0.0];
    let sim = similarity::cosine_similarity(&a, &b);
    assert_eq!(sim, 0.0);
    println!("[VERIFIED] cosine_similarity: mismatched dims = 0.0");
}

#[test]
fn test_propagation_weights() {
    // TASK-P0-001: Updated for 3-level hierarchy
    // Strategic is now top-level (was NorthStar before)
    assert_eq!(
        DefaultAlignmentCalculator::get_propagation_weight(GoalLevel::Strategic),
        1.0 // Top-level gets full weight
    );
    assert_eq!(
        DefaultAlignmentCalculator::get_propagation_weight(GoalLevel::Tactical),
        0.6 // Middle level
    );
    assert_eq!(
        DefaultAlignmentCalculator::get_propagation_weight(GoalLevel::Immediate),
        0.3 // Lowest level
    );
    println!("[VERIFIED] Propagation weights match TASK-P0-001 spec");
}
