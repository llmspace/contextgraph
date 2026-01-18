//! Helper functions for Full State Verification tests
//!
//! Contains data generation utilities and common test setup code.

use std::collections::HashSet;

use context_graph_core::types::fingerprint::{
    PurposeVector, SemanticFingerprint, SparseVector, TeleologicalFingerprint, NUM_EMBEDDERS,
};
use context_graph_storage::teleological::RocksDbTeleologicalStore;
use rand::Rng;
use tempfile::TempDir;
use uuid::Uuid;

// =============================================================================
// REAL Data Generation - NO MOCKS
// =============================================================================

pub fn generate_real_unit_vector(dim: usize) -> Vec<f32> {
    let mut rng = rand::thread_rng();
    let mut vec: Vec<f32> = (0..dim).map(|_| rng.gen_range(-1.0..1.0)).collect();
    let norm: f32 = vec.iter().map(|x| x * x).sum::<f32>().sqrt();
    if norm > f32::EPSILON {
        for v in &mut vec {
            *v /= norm;
        }
    }
    vec
}

pub fn generate_real_sparse_vector(target_nnz: usize) -> SparseVector {
    let mut rng = rand::thread_rng();
    let mut indices_set: HashSet<u16> = HashSet::new();
    while indices_set.len() < target_nnz {
        indices_set.insert(rng.gen_range(0..30522));
    }
    let mut indices: Vec<u16> = indices_set.into_iter().collect();
    indices.sort();
    let values: Vec<f32> = (0..target_nnz).map(|_| rng.gen_range(0.1..2.0)).collect();
    SparseVector::new(indices, values).expect("Failed to create sparse vector")
}

pub fn generate_real_semantic_fingerprint() -> SemanticFingerprint {
    SemanticFingerprint {
        e1_semantic: generate_real_unit_vector(1024),
        e2_temporal_recent: generate_real_unit_vector(512),
        e3_temporal_periodic: generate_real_unit_vector(512),
        e4_temporal_positional: generate_real_unit_vector(512),
        e5_causal: generate_real_unit_vector(768),
        e6_sparse: generate_real_sparse_vector(100),
        e7_code: generate_real_unit_vector(1536),
        e8_graph: generate_real_unit_vector(384),
        e9_hdc: generate_real_unit_vector(1024), // HDC projected dimension
        e10_multimodal: generate_real_unit_vector(768),
        e11_entity: generate_real_unit_vector(384),
        e12_late_interaction: vec![generate_real_unit_vector(128); 16],
        e13_splade: generate_real_sparse_vector(500),
    }
}

pub fn generate_real_purpose_vector() -> PurposeVector {
    let mut rng = rand::thread_rng();
    let mut alignments: [f32; NUM_EMBEDDERS] = [0.0; NUM_EMBEDDERS];
    for a in &mut alignments {
        *a = rng.gen_range(-1.0..1.0);
    }

    // Find dominant embedder
    let dominant_embedder = alignments
        .iter()
        .enumerate()
        .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
        .map(|(i, _)| i as u8)
        .unwrap_or(0);

    PurposeVector {
        alignments,
        dominant_embedder,
        coherence: rng.gen_range(0.0..1.0),
        stability: rng.gen_range(0.0..1.0),
    }
}

pub fn generate_real_teleological_fingerprint(_id: Uuid) -> TeleologicalFingerprint {
    TeleologicalFingerprint::new(
        generate_real_semantic_fingerprint(),
        generate_real_purpose_vector(),
        [0u8; 32],
    )
}

pub fn create_test_store(temp_dir: &TempDir) -> RocksDbTeleologicalStore {
    // Open store - EmbedderIndexRegistry is initialized in constructor
    RocksDbTeleologicalStore::open(temp_dir.path()).expect("Failed to open RocksDB store")
}

// =============================================================================
// Helper Functions
// =============================================================================

pub fn hex_string(bytes: &[u8]) -> String {
    bytes
        .iter()
        .take(64) // Limit to 64 bytes for display
        .map(|b| format!("{:02x}", b))
        .collect::<Vec<_>>()
        .join(" ")
}
