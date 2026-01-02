//! Serializable cache entry for disk persistence.
//!
//! This module provides a serializable wrapper for cache entries that avoids
//! issues with FusedEmbedding's skip_serializing_if attribute which can cause
//! bincode deserialization mismatches.

use serde::{Deserialize, Serialize};

use crate::cache::types::CacheKey;
use crate::types::dimensions::TOP_K_EXPERTS;
use crate::types::{AuxiliaryEmbeddingData, FusedEmbedding};

/// Magic bytes for cache persistence file format.
pub const CACHE_MAGIC: [u8; 4] = [0x43, 0x47, 0x45, 0x43]; // "CGEC"

/// Cache file format version.
pub const CACHE_VERSION: u8 = 1;

/// Serializable cache entry for disk persistence.
/// This avoids issues with FusedEmbedding's skip_serializing_if attribute
/// that can cause bincode deserialization mismatches.
#[derive(Serialize, Deserialize)]
pub struct SerializableCacheEntry {
    pub key: CacheKey,
    pub vector: Vec<f32>,
    pub expert_weights: [f32; 8],
    pub selected_experts: [u8; TOP_K_EXPERTS],
    pub pipeline_latency_us: u64,
    pub content_hash: u64,
    pub aux_data: Option<AuxiliaryEmbeddingData>,
}

impl SerializableCacheEntry {
    /// Create a serializable entry from a cache entry.
    pub fn from_cache_entry(key: CacheKey, embedding: &FusedEmbedding) -> Self {
        Self {
            key,
            vector: embedding.vector.clone(),
            expert_weights: embedding.expert_weights,
            selected_experts: embedding.selected_experts,
            pipeline_latency_us: embedding.pipeline_latency_us,
            content_hash: embedding.content_hash,
            aux_data: embedding.aux_data.clone(),
        }
    }

    /// Convert back to a cache key and fused embedding.
    pub fn into_key_and_embedding(self) -> (CacheKey, FusedEmbedding) {
        let embedding = FusedEmbedding {
            vector: self.vector,
            expert_weights: self.expert_weights,
            selected_experts: self.selected_experts,
            pipeline_latency_us: self.pipeline_latency_us,
            content_hash: self.content_hash,
            aux_data: self.aux_data,
        };
        (self.key, embedding)
    }
}
