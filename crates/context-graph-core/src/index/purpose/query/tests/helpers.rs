//! Test helpers for purpose query tests.

use crate::index::config::PURPOSE_VECTOR_DIM;
use crate::index::purpose::entry::{GoalId, PurposeMetadata};
use crate::types::fingerprint::PurposeVector;

/// Create a purpose vector with deterministic values based on a base value.
pub fn create_purpose_vector(base: f32) -> PurposeVector {
    let mut alignments = [0.0f32; PURPOSE_VECTOR_DIM];
    for (i, alignment) in alignments.iter_mut().enumerate() {
        *alignment = (base + i as f32 * 0.05).clamp(0.0, 1.0);
    }
    PurposeVector::new(alignments)
}

/// Create metadata for testing.
pub fn create_metadata(goal: &str) -> PurposeMetadata {
    PurposeMetadata::new(GoalId::new(goal), 0.85).unwrap()
}
