//! Tests for HnswMultiSpaceIndex.

use crate::index::config::EmbedderIndex;
use crate::index::error::IndexError;
use crate::index::hnsw_impl::HnswMultiSpaceIndex;
use crate::index::manager::MultiSpaceIndexManager;
use crate::index::status::IndexHealth;
use crate::types::fingerprint::{SemanticFingerprint, SparseVector};
use uuid::Uuid;

use super::real_hnsw_tests::random_vector;

/// Helper to create a minimal valid SemanticFingerprint.
fn create_test_fingerprint() -> SemanticFingerprint {
    SemanticFingerprint {
        e1_semantic: random_vector(1024),
        e2_temporal_recent: random_vector(512),
        e3_temporal_periodic: random_vector(512),
        e4_temporal_positional: random_vector(512),
        e5_causal: random_vector(768),
        e6_sparse: SparseVector::new(vec![100, 200], vec![0.5, 0.3]).unwrap(),
        e7_code: random_vector(1536),
        e8_graph: random_vector(384),
        e9_hdc: random_vector(1024),
        e10_multimodal: random_vector(768),
        e11_entity: random_vector(384),
        e12_late_interaction: vec![random_vector(128); 3],
        e13_splade: SparseVector::new(vec![100, 200, 300], vec![0.5, 0.3, 0.2]).unwrap(),
    }
}

#[tokio::test]
async fn test_multi_space_initialize() {
    let mut manager = HnswMultiSpaceIndex::new();

    println!("[BEFORE] Initializing MultiSpaceIndex");
    manager.initialize().await.unwrap();
    println!(
        "[AFTER] Initialized with {} indexes",
        manager.status().len()
    );

    assert_eq!(manager.status().len(), 13);

    println!("[VERIFIED] Initialize creates all 12 HNSW indexes + 1 SPLADE");
}

#[tokio::test]
async fn test_multi_space_add_vector() {
    let mut manager = HnswMultiSpaceIndex::new();
    manager.initialize().await.unwrap();

    let id = Uuid::new_v4();
    let v = random_vector(1024);

    println!("[BEFORE] Adding E1 vector");
    manager
        .add_vector(EmbedderIndex::E1Semantic, id, &v)
        .await
        .unwrap();

    let statuses = manager.status();
    let e1_status = statuses
        .iter()
        .find(|s| s.embedder == EmbedderIndex::E1Semantic)
        .unwrap();
    println!("[AFTER] E1 index has {} elements", e1_status.element_count);

    assert_eq!(e1_status.element_count, 1);
    println!("[VERIFIED] add_vector adds to correct index");
}

#[tokio::test]
async fn test_multi_space_invalid_embedder() {
    let mut manager = HnswMultiSpaceIndex::new();
    manager.initialize().await.unwrap();

    let id = Uuid::new_v4();
    let v = random_vector(100);

    println!("[BEFORE] Adding to E6Sparse (invalid for HNSW)");
    let result = manager.add_vector(EmbedderIndex::E6Sparse, id, &v).await;
    println!("[AFTER] result.is_err() = {}", result.is_err());

    assert!(matches!(result, Err(IndexError::InvalidEmbedder { .. })));
    println!("[VERIFIED] Invalid embedder rejected");
}

#[tokio::test]
async fn test_multi_space_add_fingerprint() {
    let mut manager = HnswMultiSpaceIndex::new();
    manager.initialize().await.unwrap();

    let id = Uuid::new_v4();
    let fingerprint = create_test_fingerprint();

    println!("[BEFORE] Adding complete fingerprint");
    manager.add_fingerprint(id, &fingerprint).await.unwrap();

    let statuses = manager.status();
    println!(
        "[AFTER] Status: {} indexes, total elements = {}",
        statuses.len(),
        statuses.iter().map(|s| s.element_count).sum::<usize>()
    );

    assert_eq!(statuses.len(), 13);

    let e1_status = statuses
        .iter()
        .find(|s| s.embedder == EmbedderIndex::E1Semantic)
        .unwrap();
    assert_eq!(e1_status.element_count, 1);

    let matryoshka_status = statuses
        .iter()
        .find(|s| s.embedder == EmbedderIndex::E1Matryoshka128)
        .unwrap();
    assert_eq!(matryoshka_status.element_count, 1);

    println!("[VERIFIED] add_fingerprint populates all indexes");
}

#[tokio::test]
async fn test_multi_space_search() {
    let mut manager = HnswMultiSpaceIndex::new();
    manager.initialize().await.unwrap();

    let ids: Vec<Uuid> = (0..5).map(|_| Uuid::new_v4()).collect();
    for id in &ids {
        let fp = create_test_fingerprint();
        manager.add_fingerprint(*id, &fp).await.unwrap();
    }

    println!("[BEFORE] Searching E1 semantic index");
    let query = random_vector(1024);
    let results = manager
        .search(EmbedderIndex::E1Semantic, &query, 3)
        .await
        .unwrap();
    println!(
        "[AFTER] Found {} results: {:?}",
        results.len(),
        results.iter().map(|r| r.1).collect::<Vec<_>>()
    );

    assert_eq!(results.len(), 3);
    assert!(results[0].1 >= results[1].1);
    assert!(results[1].1 >= results[2].1);

    println!("[VERIFIED] Search returns sorted results");
}

#[tokio::test]
async fn test_multi_space_search_matryoshka() {
    let mut manager = HnswMultiSpaceIndex::new();
    manager.initialize().await.unwrap();

    let id = Uuid::new_v4();
    let fp = create_test_fingerprint();
    manager.add_fingerprint(id, &fp).await.unwrap();

    println!("[BEFORE] Searching Matryoshka 128D index");
    let query_128d = random_vector(128);
    let results = manager.search_matryoshka(&query_128d, 10).await.unwrap();
    println!("[AFTER] Found {} results", results.len());

    assert_eq!(results.len(), 1);
    assert_eq!(results[0].0, id);

    println!("[VERIFIED] search_matryoshka works");
}

#[tokio::test]
async fn test_multi_space_search_purpose() {
    let mut manager = HnswMultiSpaceIndex::new();
    manager.initialize().await.unwrap();

    let id = Uuid::new_v4();
    let purpose_vec = random_vector(13);

    println!("[BEFORE] Adding and searching purpose vector");
    manager.add_purpose_vector(id, &purpose_vec).await.unwrap();

    let results = manager.search_purpose(&purpose_vec, 10).await.unwrap();
    println!(
        "[AFTER] Found {} results, top similarity = {}",
        results.len(),
        results.first().map(|r| r.1).unwrap_or(0.0)
    );

    assert_eq!(results.len(), 1);
    assert_eq!(results[0].0, id);
    assert!(results[0].1 > 0.99);

    println!("[VERIFIED] search_purpose works with high self-similarity");
}

#[tokio::test]
async fn test_multi_space_search_splade() {
    let mut manager = HnswMultiSpaceIndex::new();
    manager.initialize().await.unwrap();

    let id = Uuid::new_v4();
    let sparse = vec![(100, 0.5), (200, 0.3), (300, 0.2)];

    println!("[BEFORE] Adding and searching SPLADE");
    manager.add_splade(id, &sparse).await.unwrap();

    let results = manager.search_splade(&sparse, 10).await.unwrap();
    println!("[AFTER] Found {} results", results.len());

    assert_eq!(results.len(), 1);
    assert_eq!(results[0].0, id);

    println!("[VERIFIED] search_splade works");
}

#[tokio::test]
async fn test_multi_space_remove() {
    let mut manager = HnswMultiSpaceIndex::new();
    manager.initialize().await.unwrap();

    let id = Uuid::new_v4();
    let fp = create_test_fingerprint();
    manager.add_fingerprint(id, &fp).await.unwrap();

    let before_count: usize = manager.status().iter().map(|s| s.element_count).sum();
    println!("[BEFORE REMOVE] Total elements = {}", before_count);

    manager.remove(id).await.unwrap();

    let after_count: usize = manager.status().iter().map(|s| s.element_count).sum();
    println!("[AFTER REMOVE] Total elements = {}", after_count);

    assert_eq!(after_count, 0);
    println!("[VERIFIED] remove clears all indexes");
}

#[tokio::test]
async fn test_multi_space_not_initialized_error() {
    let manager = HnswMultiSpaceIndex::new();
    let _id = Uuid::new_v4();
    let v = random_vector(1024);

    println!("[BEFORE] Attempting search without initialization");
    let result = manager.search(EmbedderIndex::E1Semantic, &v, 10).await;
    println!("[AFTER] result.is_err() = {}", result.is_err());

    assert!(matches!(result, Err(IndexError::NotInitialized { .. })));
    println!("[VERIFIED] Operations fail before initialization");
}

#[tokio::test]
async fn test_status_returns_all_indexes() {
    let mut manager = HnswMultiSpaceIndex::new();
    manager.initialize().await.unwrap();

    let statuses = manager.status();
    println!("[STATUS] {} index statuses returned", statuses.len());

    assert_eq!(statuses.len(), 13);

    for status in &statuses {
        assert_eq!(status.health, IndexHealth::Healthy);
        println!("  {:?}: {} elements", status.embedder, status.element_count);
    }

    println!("[VERIFIED] status() returns all 13 indexes");
}
