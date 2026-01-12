//! Helper functions for creating test data (REAL data, NO mocks).

use uuid::Uuid;

use crate::index::config::PURPOSE_VECTOR_DIM;
use crate::index::purpose::entry::{GoalId, PurposeIndexEntry, PurposeMetadata};
use crate::types::fingerprint::PurposeVector;
use crate::types::JohariQuadrant;

/// Create a purpose vector with deterministic values based on base and variation.
pub fn create_purpose_vector(base: f32, variation: f32) -> PurposeVector {
    let mut alignments = [0.0f32; PURPOSE_VECTOR_DIM];
    for (i, alignment) in alignments.iter_mut().enumerate() {
        *alignment = (base + (i as f32 * variation)).clamp(0.0, 1.0);
    }
    PurposeVector::new(alignments)
}

/// Create a test entry with a specific base value and goal.
pub fn create_entry(base: f32, goal: &str) -> PurposeIndexEntry {
    let pv = create_purpose_vector(base, 0.02);
    let metadata = PurposeMetadata::new(GoalId::new(goal), 0.85, JohariQuadrant::Open).unwrap();
    PurposeIndexEntry::new(Uuid::new_v4(), pv, metadata)
}

/// Create entries forming distinct clusters.
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
