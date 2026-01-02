//! Thread-safe cache metrics with atomic counters.
//!
//! This module provides lock-free metrics tracking for cache operations
//! using atomic counters with relaxed ordering.

use std::sync::atomic::{AtomicU64, AtomicUsize, Ordering};

/// Thread-safe cache metrics with atomic counters.
///
/// All metrics use relaxed ordering since exact consistency is not required
/// for statistical monitoring.
#[derive(Debug, Default)]
pub struct CacheMetrics {
    /// Number of cache hits (key found and not expired).
    pub hits: AtomicU64,
    /// Number of cache misses (key not found or expired).
    pub misses: AtomicU64,
    /// Number of entries evicted due to capacity/memory limits.
    pub evictions: AtomicU64,
    /// Current memory usage in bytes (approximate).
    pub bytes_used: AtomicUsize,
}

impl CacheMetrics {
    /// Create new metrics with all counters at zero.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Reset all metrics to zero.
    pub fn reset(&self) {
        self.hits.store(0, Ordering::Relaxed);
        self.misses.store(0, Ordering::Relaxed);
        self.evictions.store(0, Ordering::Relaxed);
        self.bytes_used.store(0, Ordering::Relaxed);
    }

    /// Increment hit counter.
    pub fn record_hit(&self) {
        self.hits.fetch_add(1, Ordering::Relaxed);
    }

    /// Increment miss counter.
    pub fn record_miss(&self) {
        self.misses.fetch_add(1, Ordering::Relaxed);
    }

    /// Increment eviction counter.
    pub fn record_eviction(&self) {
        self.evictions.fetch_add(1, Ordering::Relaxed);
    }

    /// Add bytes to memory usage.
    pub fn add_bytes(&self, bytes: usize) {
        self.bytes_used.fetch_add(bytes, Ordering::Relaxed);
    }

    /// Subtract bytes from memory usage.
    pub fn subtract_bytes(&self, bytes: usize) {
        self.bytes_used.fetch_sub(bytes, Ordering::Relaxed);
    }
}
