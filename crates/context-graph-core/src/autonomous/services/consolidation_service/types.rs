//! Type definitions for the consolidation service.
//!
//! This module contains the core data structures used for memory consolidation:
//! - `MemoryContent` - Represents a memory with its embedding and metadata
//! - `MemoryPair` - A pair of memories to evaluate for consolidation
//! - `ServiceConsolidationCandidate` - A candidate merge with computed metrics

use crate::autonomous::curation::MemoryId;
use serde::{Deserialize, Serialize};

/// Content of a memory for consolidation purposes
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct MemoryContent {
    /// Unique identifier
    pub id: MemoryId,
    /// Embedding vector (normalized)
    pub embedding: Vec<f32>,
    /// Text content
    pub text: String,
    /// Topic alignment score
    pub alignment: f32,
    /// Access count for importance weighting
    pub access_count: u32,
}

impl MemoryContent {
    /// Create a new memory content
    pub fn new(id: MemoryId, embedding: Vec<f32>, text: String, alignment: f32) -> Self {
        Self {
            id,
            embedding,
            text,
            alignment,
            access_count: 0,
        }
    }

    /// Create with access count
    pub fn with_access_count(mut self, count: u32) -> Self {
        self.access_count = count;
        self
    }

    /// Get the embedding dimension
    pub fn dimension(&self) -> usize {
        self.embedding.len()
    }
}

/// A pair of memories to evaluate for consolidation
#[derive(Clone, Debug)]
pub struct MemoryPair {
    /// First memory
    pub a: MemoryContent,
    /// Second memory
    pub b: MemoryContent,
}

impl MemoryPair {
    /// Create a new memory pair
    pub fn new(a: MemoryContent, b: MemoryContent) -> Self {
        Self { a, b }
    }

    /// Get alignment difference between the two memories
    pub fn alignment_diff(&self) -> f32 {
        (self.a.alignment - self.b.alignment).abs()
    }
}

/// Candidate for consolidation with computed metrics
#[derive(Clone, Debug)]
pub struct ServiceConsolidationCandidate {
    /// Source memory IDs to merge
    pub source_ids: Vec<MemoryId>,
    /// Target memory ID (result of merge)
    pub target_id: MemoryId,
    /// Similarity score between sources
    pub similarity: f32,
    /// Combined alignment of merged memory
    pub combined_alignment: f32,
}

impl ServiceConsolidationCandidate {
    /// Create a new consolidation candidate
    pub fn new(
        source_ids: Vec<MemoryId>,
        target_id: MemoryId,
        similarity: f32,
        combined_alignment: f32,
    ) -> Self {
        Self {
            source_ids,
            target_id,
            similarity,
            combined_alignment,
        }
    }
}
