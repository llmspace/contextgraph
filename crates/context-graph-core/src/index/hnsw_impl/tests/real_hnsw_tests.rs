//! Tests for RealHnswIndex.

use crate::index::config::{DistanceMetric, HnswConfig};
use crate::index::error::IndexError;
use crate::index::hnsw_impl::RealHnswIndex;
use uuid::Uuid;

/// Helper to create a random normalized vector.
pub(super) fn random_vector(dim: usize) -> Vec<f32> {
    use rand::Rng;
    let mut rng = rand::thread_rng();
    let mut v: Vec<f32> = (0..dim).map(|_| rng.gen_range(-1.0..1.0)).collect();

    let norm: f32 = v.iter().map(|x| x * x).sum::<f32>().sqrt();
    for x in &mut v {
        *x /= norm;
    }
    v
}

#[test]
fn test_real_hnsw_new() {
    let config = HnswConfig::default_for_dimension(1024, DistanceMetric::Cosine);
    let index = RealHnswIndex::new(config).expect("Failed to create index");

    assert_eq!(index.len(), 0);
    assert!(index.is_empty());
    println!("[VERIFIED] RealHnswIndex::new() creates empty index");
}

#[test]
fn test_real_hnsw_add_and_search() {
    let config = HnswConfig::default_for_dimension(128, DistanceMetric::Cosine);
    let mut index = RealHnswIndex::new(config).expect("Failed to create index");

    let id1 = Uuid::new_v4();
    let id2 = Uuid::new_v4();
    let v1 = random_vector(128);
    let v2 = random_vector(128);

    println!("[BEFORE] Adding 2 vectors to HNSW");
    index.add(id1, &v1).unwrap();
    index.add(id2, &v2).unwrap();
    println!("[AFTER] index.len() = {}", index.len());

    assert_eq!(index.len(), 2);

    let results = index.search(&v1, 2).unwrap();
    println!(
        "[SEARCH] Found {} results, top result = {:?}",
        results.len(),
        results.first()
    );

    assert_eq!(results.len(), 2);
    assert_eq!(results[0].0, id1);
    assert!(results[0].1 > results[1].1);

    println!("[VERIFIED] Add and search work correctly");
}

#[test]
fn test_real_hnsw_dimension_mismatch() {
    let config = HnswConfig::default_for_dimension(128, DistanceMetric::Cosine);
    let mut index = RealHnswIndex::new(config).expect("Failed to create index");

    let id = Uuid::new_v4();
    let wrong_dim = random_vector(256);

    println!("[BEFORE] Adding vector with wrong dimension (256 vs 128)");
    let result = index.add(id, &wrong_dim);
    println!("[AFTER] result.is_err() = {}", result.is_err());

    assert!(matches!(
        result,
        Err(IndexError::DimensionMismatch {
            expected: 128,
            actual: 256,
            ..
        })
    ));
    println!("[VERIFIED] Dimension mismatch rejected");
}

#[test]
fn test_real_hnsw_zero_norm_rejected() {
    let config = HnswConfig::default_for_dimension(10, DistanceMetric::Cosine);
    let mut index = RealHnswIndex::new(config).expect("Failed to create index");

    let id = Uuid::new_v4();
    let zero_vec = vec![0.0; 10];

    println!("[BEFORE] Adding zero-norm vector");
    let result = index.add(id, &zero_vec);
    println!("[AFTER] result = {:?}", result.is_err());

    assert!(matches!(result, Err(IndexError::ZeroNormVector { .. })));
    println!("[VERIFIED] Zero-norm vector rejected");
}

#[test]
fn test_real_hnsw_remove() {
    let config = HnswConfig::default_for_dimension(64, DistanceMetric::Cosine);
    let mut index = RealHnswIndex::new(config).expect("Failed to create index");

    let id = Uuid::new_v4();
    let v = random_vector(64);

    index.add(id, &v).unwrap();
    println!("[BEFORE REMOVE] index.len() = {}", index.len());

    let removed = index.remove(id);
    println!(
        "[AFTER REMOVE] index.len() = {}, removed = {}",
        index.len(),
        removed
    );

    assert!(removed);
    assert_eq!(index.len(), 0);
    println!("[VERIFIED] Remove works correctly");
}

#[test]
fn test_memory_usage_calculation() {
    let config = HnswConfig::default_for_dimension(1024, DistanceMetric::Cosine);
    let mut index = RealHnswIndex::new(config).expect("Failed to create index");

    let empty_usage = index.memory_usage();
    println!("[BEFORE] Empty index memory: {} bytes", empty_usage);

    for _ in 0..100 {
        let id = Uuid::new_v4();
        let v = random_vector(1024);
        index.add(id, &v).unwrap();
    }

    let full_usage = index.memory_usage();
    println!("[AFTER] 100 vectors memory: {} bytes", full_usage);

    assert!(full_usage > empty_usage);
    assert!(full_usage > 400_000);

    println!("[VERIFIED] Memory usage calculation reasonable");
}
