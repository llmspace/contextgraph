//! Tests for RocksDbTeleologicalStore.
//!
//! Comprehensive tests for CRUD operations, persistence, and trait compliance.

#![cfg(test)]

use super::*;
use tempfile::TempDir;

use context_graph_core::traits::TeleologicalMemoryStore;
use context_graph_core::types::fingerprint::{
    JohariFingerprint, PurposeVector, SemanticFingerprint, SparseVector, TeleologicalFingerprint,
    NUM_EMBEDDERS,
};

/// Create a test fingerprint with real (non-zero) embeddings.
/// Uses deterministic pseudo-random values seeded from a counter.
fn create_test_fingerprint_with_seed(seed: u64) -> TeleologicalFingerprint {
    use std::f32::consts::PI;

    // Generate deterministic embeddings from seed
    let generate_vec = |dim: usize, s: u64| -> Vec<f32> {
        (0..dim)
            .map(|i| {
                let x = ((s as f64 * 0.1 + i as f64 * 0.01) * PI as f64).sin() as f32;
                x * 0.5 + 0.5 // Normalize to [0, 1] range
            })
            .collect()
    };

    // Generate deterministic sparse vector for SPLADE
    let generate_sparse = |s: u64| -> SparseVector {
        let num_entries = 50 + (s % 50) as usize;
        let mut indices: Vec<u16> = Vec::with_capacity(num_entries);
        let mut values: Vec<f32> = Vec::with_capacity(num_entries);
        for i in 0..num_entries {
            let idx = ((s + i as u64 * 31) % 30522) as u16; // u16 for sparse indices
            let val = ((s as f64 * 0.1 + i as f64 * 0.2) * PI as f64).sin().abs() as f32 + 0.1;
            indices.push(idx);
            values.push(val);
        }
        SparseVector { indices, values }
    };

    // Generate late-interaction vectors (variable number of 128D token vectors)
    let generate_late_interaction = |s: u64| -> Vec<Vec<f32>> {
        let num_tokens = 5 + (s % 10) as usize;
        (0..num_tokens)
            .map(|t| generate_vec(128, s + t as u64 * 100))
            .collect()
    };

    // Create SemanticFingerprint with correct fields (per semantic/fingerprint.rs)
    let semantic = SemanticFingerprint {
        e1_semantic: generate_vec(1024, seed),               // 1024D
        e2_temporal_recent: generate_vec(512, seed + 1),     // 512D
        e3_temporal_periodic: generate_vec(512, seed + 2),   // 512D
        e4_temporal_positional: generate_vec(512, seed + 3), // 512D
        e5_causal: generate_vec(768, seed + 4),              // 768D
        e6_sparse: generate_sparse(seed + 5),                // Sparse
        e7_code: generate_vec(1536, seed + 6),               // 1536D
        e8_graph: generate_vec(384, seed + 7),               // 384D
        e9_hdc: generate_vec(1024, seed + 8),                // 1024D HDC (projected)
        e10_multimodal: generate_vec(768, seed + 9),         // 768D
        e11_entity: generate_vec(384, seed + 10),            // 384D
        e12_late_interaction: generate_late_interaction(seed + 11), // Vec<Vec<f32>>
        e13_splade: generate_sparse(seed + 12),              // Sparse
    };

    // Create PurposeVector with correct structure (alignments: [f32; 13])
    let mut alignments = [0.5f32; NUM_EMBEDDERS];
    for (i, a) in alignments.iter_mut().enumerate() {
        *a = ((seed as f64 * 0.1 + i as f64 * 0.3) * std::f32::consts::PI as f64).sin() as f32
            * 0.5
            + 0.5;
    }
    let purpose = PurposeVector::new(alignments);

    // Create JohariFingerprint using zeroed() then set values
    let mut johari = JohariFingerprint::zeroed();
    for i in 0..NUM_EMBEDDERS {
        johari.set_quadrant(i, 0.8, 0.1, 0.05, 0.05, 0.9); // Open dominant with high confidence
    }

    // Create unique hash
    let mut hash = [0u8; 32];
    for (i, byte) in hash.iter_mut().enumerate() {
        *byte = ((seed + i as u64) % 256) as u8;
    }

    TeleologicalFingerprint::new(semantic, purpose, johari, hash)
}

fn create_test_fingerprint() -> TeleologicalFingerprint {
    create_test_fingerprint_with_seed(42)
}

/// Helper to create store with initialized indexes.
///
/// Note: EmbedderIndexRegistry is initialized in the constructor,
/// so no separate initialization step is needed.
fn create_initialized_store(path: &std::path::Path) -> RocksDbTeleologicalStore {
    RocksDbTeleologicalStore::open(path).unwrap()
}

#[tokio::test]
async fn test_open_and_health_check() {
    let tmp = TempDir::new().unwrap();
    let store = create_initialized_store(tmp.path());
    assert!(store.health_check().is_ok());
}

#[tokio::test]
async fn test_store_and_retrieve() {
    let tmp = TempDir::new().unwrap();
    let store = create_initialized_store(tmp.path());

    let fp = create_test_fingerprint();
    let id = fp.id;

    // Store
    let stored_id = store.store(fp.clone()).await.unwrap();
    assert_eq!(stored_id, id);

    // Retrieve
    let retrieved = store.retrieve(id).await.unwrap();
    assert!(retrieved.is_some());
    let retrieved_fp = retrieved.unwrap();
    assert_eq!(retrieved_fp.id, id);
}

#[tokio::test]
async fn test_physical_persistence() {
    let tmp = TempDir::new().unwrap();
    let path = tmp.path().to_path_buf();

    let fp = create_test_fingerprint();
    let id = fp.id;

    // Store and close
    {
        let store = create_initialized_store(&path);
        store.store(fp.clone()).await.unwrap();
        store.flush().await.unwrap();
    }

    // Reopen and verify
    {
        let store = create_initialized_store(&path);
        let retrieved = store.retrieve(id).await.unwrap();
        assert!(
            retrieved.is_some(),
            "Fingerprint should persist across database close/reopen"
        );
        assert_eq!(retrieved.unwrap().id, id);
    }

    // Verify raw bytes exist in RocksDB
    {
        let store = create_initialized_store(&path);
        let raw = store.get_fingerprint_raw(id).unwrap();
        assert!(raw.is_some(), "Raw bytes should exist in RocksDB");
        let raw_bytes = raw.unwrap();
        // With E9_DIM = 1024 (projected), fingerprints are ~32-40KB
        assert!(
            raw_bytes.len() >= 25000,
            "Serialized fingerprint should be >= 25KB, got {} bytes",
            raw_bytes.len()
        );
    }
}

#[tokio::test]
async fn test_delete_soft() {
    let tmp = TempDir::new().unwrap();
    let store = create_initialized_store(tmp.path());

    let fp = create_test_fingerprint();
    let id = fp.id;

    store.store(fp).await.unwrap();
    let deleted = store.delete(id, true).await.unwrap();
    assert!(deleted);

    // Should not be retrievable after soft delete
    let retrieved = store.retrieve(id).await.unwrap();
    assert!(retrieved.is_none());
}

#[tokio::test]
async fn test_delete_hard() {
    let tmp = TempDir::new().unwrap();
    let store = create_initialized_store(tmp.path());

    let fp = create_test_fingerprint();
    let id = fp.id;

    store.store(fp).await.unwrap();
    let deleted = store.delete(id, false).await.unwrap();
    assert!(deleted);

    // Raw bytes should be gone
    let raw = store.get_fingerprint_raw(id).unwrap();
    assert!(raw.is_none());
}

#[tokio::test]
async fn test_count() {
    let tmp = TempDir::new().unwrap();
    let store = create_initialized_store(tmp.path());

    assert_eq!(store.count().await.unwrap(), 0);

    store
        .store(create_test_fingerprint_with_seed(1))
        .await
        .unwrap();
    store
        .store(create_test_fingerprint_with_seed(2))
        .await
        .unwrap();
    store
        .store(create_test_fingerprint_with_seed(3))
        .await
        .unwrap();

    assert_eq!(store.count().await.unwrap(), 3);
}

#[tokio::test]
async fn test_backend_type() {
    let tmp = TempDir::new().unwrap();
    let store = create_initialized_store(tmp.path());
    assert_eq!(
        store.backend_type(),
        context_graph_core::traits::TeleologicalStorageBackend::RocksDb
    );
}
