//! Helper functions for Purpose Index tests - REAL DATA ONLY (NO MOCKS)

use crate::index::config::{DistanceMetric, HnswConfig, PURPOSE_VECTOR_DIM};
use crate::index::purpose::entry::{GoalId, PurposeIndexEntry, PurposeMetadata};
use crate::types::fingerprint::PurposeVector;
use uuid::Uuid;

/// Create a purpose vector with deterministic values based on base and variation.
/// Uses REAL alignment values in range [0.0, 1.0].
pub fn create_purpose_vector(base: f32, variation: f32) -> PurposeVector {
    let mut alignments = [0.0f32; PURPOSE_VECTOR_DIM];
    for (i, alignment) in alignments.iter_mut().enumerate() {
        *alignment = (base + (i as f32 * variation)).clamp(0.0, 1.0);
    }
    PurposeVector::new(alignments)
}

/// Create metadata with a specific goal.
pub fn create_metadata(goal: &str) -> PurposeMetadata {
    PurposeMetadata::new(GoalId::new(goal), 0.85).unwrap()
}

/// Create a complete purpose index entry with real data.
pub fn create_entry(base: f32, goal: &str) -> PurposeIndexEntry {
    let pv = create_purpose_vector(base, 0.02);
    let metadata = create_metadata(goal);
    PurposeIndexEntry::new(Uuid::new_v4(), pv, metadata)
}

/// Create a purpose index entry with a specific memory ID.
pub fn create_entry_with_id(memory_id: Uuid, base: f32, goal: &str) -> PurposeIndexEntry {
    let pv = create_purpose_vector(base, 0.02);
    let metadata = create_metadata(goal);
    PurposeIndexEntry::new(memory_id, pv, metadata)
}

/// Create a default HNSW config for purpose vectors.
pub fn purpose_config() -> HnswConfig {
    HnswConfig::new(16, 200, 100, DistanceMetric::Cosine, PURPOSE_VECTOR_DIM)
}

/// Create entries forming distinct clusters for clustering tests.
pub fn create_clustered_entries() -> Vec<PurposeIndexEntry> {
    let mut entries = Vec::new();

    // Cluster 1: low values (base around 0.2)
    for i in 0..5 {
        entries.push(create_entry(0.15 + i as f32 * 0.02, "goal_low"));
    }

    // Cluster 2: medium values (base around 0.5)
    for i in 0..5 {
        entries.push(create_entry(0.45 + i as f32 * 0.02, "goal_mid"));
    }

    // Cluster 3: high values (base around 0.8)
    for i in 0..5 {
        entries.push(create_entry(0.75 + i as f32 * 0.02, "goal_high"));
    }

    entries
}
