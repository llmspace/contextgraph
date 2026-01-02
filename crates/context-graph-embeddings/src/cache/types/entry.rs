//! Cache entry type for the embedding cache system.
//!
//! This module provides [`CacheEntry`], a cached embedding with LRU/LFU metadata.

use std::sync::atomic::{AtomicU32, AtomicU64, Ordering};
use std::time::{Duration, Instant};

use once_cell::sync::Lazy;

use crate::types::FusedEmbedding;

/// Process start instant for relative timestamp storage.
/// Using nanos since start allows compact u64 atomic storage.
pub(crate) static START_INSTANT: Lazy<Instant> = Lazy::new(Instant::now);

/// Metadata size for CacheEntry (Instant + AtomicU64 + AtomicU32).
const CACHE_ENTRY_METADATA_SIZE: usize = 16 + 8 + 4;

/// Cached embedding with LRU/LFU metadata.
///
/// # Memory Layout (estimated)
/// - FusedEmbedding: ~6198 bytes (1536 f32s + metadata)
/// - Instant: 16 bytes
/// - AtomicU64: 8 bytes
/// - AtomicU32: 4 bytes
/// - Total: ~6226 bytes per entry
///
/// # Thread Safety
/// - `last_accessed` and `access_count` use atomics for lock-free updates
/// - `embedding` is read-only after creation
#[derive(Debug)]
pub struct CacheEntry {
    /// The cached fused embedding (immutable after creation)
    pub embedding: FusedEmbedding,
    /// Creation timestamp for TTL expiration
    created_at: Instant,
    /// Last access time as nanos since process start (for LRU)
    last_accessed: AtomicU64,
    /// Access count (for LFU)
    access_count: AtomicU32,
}

impl CacheEntry {
    /// Create new cache entry with current timestamp.
    /// Sets `last_accessed` to now, `access_count` to 1.
    #[must_use]
    pub fn new(embedding: FusedEmbedding) -> Self {
        let now = START_INSTANT.elapsed().as_nanos() as u64;
        Self {
            embedding,
            created_at: Instant::now(),
            last_accessed: AtomicU64::new(now),
            access_count: AtomicU32::new(1),
        }
    }

    /// Update last_accessed timestamp (for LRU policy).
    /// Uses Ordering::Relaxed - eventual consistency is acceptable.
    pub fn touch(&self) {
        let now = START_INSTANT.elapsed().as_nanos() as u64;
        self.last_accessed.store(now, Ordering::Relaxed);
    }

    /// Increment access count (for LFU policy).
    /// Uses Ordering::Relaxed.
    pub fn increment_access(&self) {
        self.access_count.fetch_add(1, Ordering::Relaxed);
    }

    /// Get current access count.
    #[must_use]
    pub fn access_count(&self) -> u32 {
        self.access_count.load(Ordering::Relaxed)
    }

    /// Time since creation.
    #[must_use]
    pub fn age(&self) -> Duration {
        self.created_at.elapsed()
    }

    /// Check if entry has expired based on TTL.
    #[must_use]
    pub fn is_expired(&self, ttl: Duration) -> bool {
        self.age() >= ttl
    }

    /// Total memory size in bytes (for max_bytes budget).
    /// Returns: embedding.memory_size() + sizeof(metadata)
    #[must_use]
    pub fn memory_size(&self) -> usize {
        self.embedding.memory_size() + CACHE_ENTRY_METADATA_SIZE
    }

    /// Get creation timestamp.
    #[must_use]
    pub fn created_at(&self) -> Instant {
        self.created_at
    }

    /// Get last access time as duration since process start.
    #[must_use]
    pub fn last_accessed(&self) -> Duration {
        let nanos = self.last_accessed.load(Ordering::Relaxed);
        Duration::from_nanos(nanos)
    }
}

#[cfg(test)]
#[path = "entry_tests.rs"]
mod tests;
