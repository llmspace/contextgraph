//! Purpose search result types.
//!
//! This module provides [`PurposeSearchResult`] which represents a single
//! result from a purpose-based search operation.

use uuid::Uuid;

use crate::types::fingerprint::PurposeVector;

use super::super::entry::{GoalId, PurposeMetadata};

/// Result from a purpose-based search operation.
///
/// Contains the matching memory's ID, similarity score, purpose vector,
/// and associated metadata.
///
/// # Fields
///
/// - `memory_id`: UUID of the matching memory
/// - `purpose_similarity`: Similarity score in purpose space [0.0, 1.0]
/// - `purpose_vector`: The full 13D purpose vector
/// - `metadata`: Associated metadata (goal, confidence)
///
/// # Ordering
///
/// Results are typically ordered by `purpose_similarity` (descending).
#[derive(Clone, Debug)]
pub struct PurposeSearchResult {
    /// The matching memory ID.
    pub memory_id: Uuid,

    /// Similarity score in purpose space.
    ///
    /// Range: [0.0, 1.0] where 1.0 is identical purpose alignment.
    pub purpose_similarity: f32,

    /// The purpose vector of the matching memory.
    pub purpose_vector: PurposeVector,

    /// Associated metadata about the purpose computation.
    pub metadata: PurposeMetadata,
}

impl PurposeSearchResult {
    /// Create a new search result.
    ///
    /// # Arguments
    ///
    /// * `memory_id` - UUID of the matching memory
    /// * `purpose_similarity` - Similarity score [0.0, 1.0]
    /// * `purpose_vector` - The 13D purpose vector
    /// * `metadata` - Associated metadata
    pub fn new(
        memory_id: Uuid,
        purpose_similarity: f32,
        purpose_vector: PurposeVector,
        metadata: PurposeMetadata,
    ) -> Self {
        Self {
            memory_id,
            purpose_similarity,
            purpose_vector,
            metadata,
        }
    }

    /// Get the aggregate alignment of this result's purpose vector.
    #[inline]
    pub fn aggregate_alignment(&self) -> f32 {
        self.purpose_vector.aggregate_alignment()
    }

    /// Get the dominant embedder index.
    #[inline]
    pub fn dominant_embedder(&self) -> u8 {
        self.purpose_vector.dominant_embedder
    }

    /// Get the coherence score.
    #[inline]
    pub fn coherence(&self) -> f32 {
        self.purpose_vector.coherence
    }

    /// Check if this result passes a given goal filter.
    #[inline]
    pub fn matches_goal(&self, goal: &GoalId) -> bool {
        &self.metadata.primary_goal == goal
    }
}
