//! Types and constants for the Reflex layer.
//!
//! This module contains the core types used by the Modern Hopfield Network cache
//! and the Reflex layer.

use serde::{Deserialize, Serialize};

// ============================================================
// Constants
// ============================================================

/// Default cache capacity (practical limit for <100us lookup)
pub const DEFAULT_CACHE_CAPACITY: usize = 10_000;

/// Default inverse temperature (beta) for Hopfield retrieval.
/// Higher values = sharper retrieval (more winner-take-all).
pub const DEFAULT_BETA: f32 = 1.0;

/// Minimum similarity threshold for a cache hit.
/// Below this, we consider it a miss even if patterns exist.
pub const MIN_HIT_SIMILARITY: f32 = 0.85;

/// Embedding dimension for cache patterns.
/// Must match the embedding dimension from L1 Sensing output.
pub const PATTERN_DIM: usize = 128;

// ============================================================
// CachedResponse - What we store and retrieve
// ============================================================

/// A cached response stored in the Hopfield network.
///
/// This represents a pattern-response pair: given a query pattern similar
/// to the stored pattern, return this response.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CachedResponse {
    /// Unique identifier for this cached entry
    pub id: String,

    /// The stored pattern vector (key)
    pub pattern: Vec<f32>,

    /// The response data (value)
    pub response_data: serde_json::Value,

    /// Confidence score of this cached response [0, 1]
    pub confidence: f32,

    /// Number of times this pattern has been accessed
    pub access_count: u64,

    /// Timestamp when this entry was created (Unix epoch millis)
    pub created_at: u64,

    /// Timestamp of last access (Unix epoch millis)
    pub last_accessed: u64,
}

impl CachedResponse {
    /// Create a new cached response.
    pub fn new(
        id: String,
        pattern: Vec<f32>,
        response_data: serde_json::Value,
        confidence: f32,
    ) -> Self {
        let now = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap_or_default()
            .as_millis() as u64;

        Self {
            id,
            pattern,
            response_data,
            confidence,
            access_count: 0,
            created_at: now,
            last_accessed: now,
        }
    }
}

// ============================================================
// Cache Statistics
// ============================================================

/// Statistics about the Hopfield cache.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CacheStats {
    /// Total cache hits
    pub hit_count: u64,

    /// Total cache misses
    pub miss_count: u64,

    /// Number of patterns currently stored
    pub patterns_stored: usize,

    /// Maximum capacity
    pub capacity: usize,

    /// Average retrieval time in microseconds
    pub avg_retrieval_us: f64,

    /// Current beta (inverse temperature)
    pub beta: f32,
}

impl CacheStats {
    /// Calculate hit rate as a percentage.
    pub fn hit_rate(&self) -> f64 {
        let total = self.hit_count + self.miss_count;
        if total == 0 {
            0.0
        } else {
            (self.hit_count as f64 / total as f64) * 100.0
        }
    }
}
