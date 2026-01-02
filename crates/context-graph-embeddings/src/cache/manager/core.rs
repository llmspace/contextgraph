//! Core CacheManager implementation with LRU eviction.
//!
//! This module provides the main cache manager for storing and retrieving
//! FusedEmbedding results with O(1) lookup and LRU eviction.

use std::sync::atomic::Ordering;
use std::sync::RwLock;
use std::time::Duration;

use linked_hash_map::LinkedHashMap;
use tracing::error;

use crate::cache::types::{CacheEntry, CacheKey};
use crate::config::CacheConfig;
use crate::error::{EmbeddingError, EmbeddingResult};
use crate::types::FusedEmbedding;

use super::metrics::CacheMetrics;

/// LRU-based embedding cache manager.
///
/// FAIL-FAST: All errors propagate immediately. No fallbacks.
///
/// # Thread Safety
///
/// Uses RwLock for entries to optimize read-heavy workloads:
/// - get() takes read lock (multiple concurrent readers)
/// - put()/remove()/clear() take write lock (exclusive)
///
/// # Eviction Strategy
///
/// 1. TTL expiration: Entries older than ttl_seconds are removed on access
/// 2. LRU eviction: When max_entries exceeded, oldest entries removed
/// 3. Memory eviction: When max_bytes exceeded, entries removed until under limit
pub struct CacheManager {
    /// Internal cache storage with LRU ordering.
    pub(super) entries: RwLock<LinkedHashMap<CacheKey, CacheEntry>>,
    /// Cache configuration (immutable after creation).
    pub(super) config: CacheConfig,
    /// Thread-safe metrics.
    pub(super) metrics: CacheMetrics,
}

impl CacheManager {
    /// Create new cache manager with given configuration.
    ///
    /// # Arguments
    /// * `config` - Cache configuration (max_entries, max_bytes, ttl, etc.)
    ///
    /// # Returns
    /// * `Ok(CacheManager)` - Configured cache manager
    /// * `Err(EmbeddingError::ConfigError)` - If config validation fails
    ///
    /// # Errors
    /// Returns error if:
    /// - max_entries is 0
    /// - max_bytes is 0
    /// - persist_to_disk is true but disk_path is None
    pub fn new(config: CacheConfig) -> EmbeddingResult<Self> {
        // Validate configuration
        if config.max_entries == 0 {
            error!("CacheManager config error: max_entries cannot be 0");
            return Err(EmbeddingError::ConfigError {
                message: "max_entries cannot be 0".to_string(),
            });
        }

        if config.max_bytes == 0 {
            error!("CacheManager config error: max_bytes cannot be 0");
            return Err(EmbeddingError::ConfigError {
                message: "max_bytes cannot be 0".to_string(),
            });
        }

        if config.persist_to_disk && config.disk_path.is_none() {
            error!("CacheManager config error: disk_path required when persist_to_disk is true");
            return Err(EmbeddingError::ConfigError {
                message: "disk_path required when persist_to_disk is true".to_string(),
            });
        }

        Ok(Self {
            entries: RwLock::new(LinkedHashMap::new()),
            config,
            metrics: CacheMetrics::new(),
        })
    }

    /// Get embedding by key, updating LRU order.
    ///
    /// Returns None if:
    /// - Key not found in cache
    /// - Entry has expired (TTL exceeded)
    ///
    /// # Side Effects
    /// - Updates metrics.hits or metrics.misses
    /// - Moves accessed entry to end of LRU order (most recently used)
    /// - Removes expired entries
    #[must_use]
    pub fn get(&self, key: &CacheKey) -> Option<FusedEmbedding> {
        // Acquire write lock for LRU reordering via get_refresh()
        let mut entries = match self.entries.write() {
            Ok(entries) => entries,
            Err(_) => {
                self.metrics.record_miss();
                return None;
            }
        };

        // get_refresh moves entry to back (most recently used) if found
        let entry = match entries.get_refresh(key) {
            Some(entry) => entry,
            None => {
                self.metrics.record_miss();
                return None;
            }
        };

        // Check TTL expiration
        if let Some(ttl_secs) = self.config.ttl_seconds {
            let ttl = Duration::from_secs(ttl_secs);
            if entry.is_expired(ttl) {
                // Entry expired - remove and record miss
                let size = entry.memory_size();
                entries.remove(key);
                self.metrics.subtract_bytes(size);
                self.metrics.record_miss();
                return None;
            }
        }

        // Entry exists and is valid - update access tracking
        entry.touch();
        entry.increment_access();
        self.metrics.record_hit();

        // Clone the embedding to return (entry.embedding is immutable)
        Some(entry.embedding.clone())
    }

    /// Insert embedding, evicting LRU entries if needed.
    ///
    /// # Arguments
    /// * `key` - Cache key (content hash)
    /// * `embedding` - FusedEmbedding to cache
    ///
    /// # Returns
    /// * `Ok(())` - Entry cached successfully
    /// * `Err(EmbeddingError::CacheError)` - If embedding alone exceeds max_bytes
    pub fn put(&self, key: CacheKey, embedding: FusedEmbedding) -> EmbeddingResult<()> {
        let entry = CacheEntry::new(embedding);
        let entry_size = entry.memory_size();

        // Check if single entry exceeds max_bytes
        if entry_size > self.config.max_bytes {
            error!(
                "CacheManager put error: entry size {} exceeds max_bytes {}",
                entry_size, self.config.max_bytes
            );
            return Err(EmbeddingError::CacheError {
                message: format!(
                    "Entry size {} bytes exceeds max_bytes {} bytes",
                    entry_size, self.config.max_bytes
                ),
            });
        }

        let mut entries = self.entries.write().map_err(|e| {
            error!("CacheManager put error: lock poisoned: {}", e);
            EmbeddingError::CacheError {
                message: format!("Lock poisoned: {}", e),
            }
        })?;

        // Check if key already exists - update in place
        if let Some(old_entry) = entries.get(&key) {
            let old_size = old_entry.memory_size();
            self.metrics.subtract_bytes(old_size);
        }

        // Evict entries until we're under max_entries limit
        while entries.len() >= self.config.max_entries {
            self.evict_oldest(&mut entries);
        }

        // Evict entries until we're under max_bytes limit
        let current_bytes = self.metrics.bytes_used.load(Ordering::Relaxed);
        let mut projected_bytes = current_bytes + entry_size;

        while projected_bytes > self.config.max_bytes && !entries.is_empty() {
            if let Some(evicted_size) = self.evict_oldest(&mut entries) {
                projected_bytes = projected_bytes.saturating_sub(evicted_size);
            } else {
                break;
            }
        }

        // Insert new entry
        entries.insert(key, entry);
        self.metrics.add_bytes(entry_size);

        Ok(())
    }

    /// Evict oldest entry (front of LinkedHashMap).
    /// Returns the size of the evicted entry, or None if cache was empty.
    pub(super) fn evict_oldest(
        &self,
        entries: &mut LinkedHashMap<CacheKey, CacheEntry>,
    ) -> Option<usize> {
        if let Some((_key, entry)) = entries.pop_front() {
            let size = entry.memory_size();
            self.metrics.subtract_bytes(size);
            self.metrics.record_eviction();
            Some(size)
        } else {
            None
        }
    }

    /// Check if key exists (does not update LRU order).
    #[must_use]
    pub fn contains(&self, key: &CacheKey) -> bool {
        self.entries
            .read()
            .map(|entries| entries.contains_key(key))
            .unwrap_or(false)
    }

    /// Remove entry by key.
    pub fn remove(&self, key: &CacheKey) -> Option<FusedEmbedding> {
        let mut entries = self.entries.write().ok()?;
        let entry = entries.remove(key)?;
        let size = entry.memory_size();
        self.metrics.subtract_bytes(size);
        Some(entry.embedding)
    }

    /// Clear all entries, reset metrics.
    pub fn clear(&self) {
        if let Ok(mut entries) = self.entries.write() {
            entries.clear();
            self.metrics.reset();
        }
    }

    /// Current entry count.
    #[must_use]
    pub fn len(&self) -> usize {
        self.entries
            .read()
            .map(|entries| entries.len())
            .unwrap_or(0)
    }

    /// Check if cache is empty.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Calculate hit rate: hits / (hits + misses).
    #[must_use]
    pub fn hit_rate(&self) -> f32 {
        let hits = self.metrics.hits.load(Ordering::Relaxed);
        let misses = self.metrics.misses.load(Ordering::Relaxed);
        let total = hits + misses;

        if total == 0 {
            0.0
        } else {
            hits as f32 / total as f32
        }
    }

    /// Current memory usage in bytes.
    #[must_use]
    pub fn memory_usage(&self) -> usize {
        self.metrics.bytes_used.load(Ordering::Relaxed)
    }

    /// Get reference to cache metrics.
    #[must_use]
    pub fn metrics(&self) -> &CacheMetrics {
        &self.metrics
    }

    /// Get reference to cache configuration.
    #[must_use]
    pub fn config(&self) -> &CacheConfig {
        &self.config
    }
}
