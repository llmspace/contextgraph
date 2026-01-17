//! Modern Hopfield Network Cache implementation.
//!
//! Uses continuous-valued attention-based retrieval (softmax over patterns)
//! for instant lookup of cached responses.

use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::RwLock;
use std::time::Instant;

use crate::error::{CoreError, CoreResult};

use super::math::dot_product_f32;
#[allow(deprecated)]
use super::types::MIN_HIT_SIMILARITY;
use super::types::{CacheStats, CachedResponse, DEFAULT_BETA, DEFAULT_CACHE_CAPACITY, PATTERN_DIM};

// ============================================================
// Modern Hopfield Network Cache
// ============================================================

/// Modern Hopfield Network for associative memory cache.
///
/// Uses continuous-valued attention-based retrieval (softmax over patterns)
/// for instant lookup of cached responses.
///
/// # Formula
///
/// ```text
/// attention = softmax(beta * patterns^T * query)
/// output = attention * patterns
/// ```
///
/// The softmax ensures that similar patterns contribute more to the output,
/// providing automatic pattern completion and noise tolerance.
#[derive(Debug)]
pub struct ModernHopfieldCache {
    /// Stored pattern-response pairs
    entries: RwLock<Vec<CachedResponse>>,

    /// Maximum number of patterns to store
    pub(crate) capacity: usize,

    /// Inverse temperature for retrieval sharpness
    beta: f32,

    /// Total retrieval time for averaging (in microseconds)
    total_retrieval_us: AtomicU64,

    /// Number of retrievals for averaging
    retrieval_count: AtomicU64,
}

impl ModernHopfieldCache {
    /// Create a new Modern Hopfield cache with default settings.
    pub fn new() -> Self {
        Self::with_capacity(DEFAULT_CACHE_CAPACITY)
    }

    /// Create a cache with specified capacity.
    pub fn with_capacity(capacity: usize) -> Self {
        Self {
            entries: RwLock::new(Vec::with_capacity(capacity.min(10_000))),
            capacity,
            beta: DEFAULT_BETA,
            total_retrieval_us: AtomicU64::new(0),
            retrieval_count: AtomicU64::new(0),
        }
    }

    /// Create a cache with custom beta (inverse temperature).
    pub fn with_beta(mut self, beta: f32) -> Self {
        self.beta = beta.clamp(0.1, 10.0); // Clamp to reasonable range
        self
    }

    /// Check if the cache is ready for operations.
    pub fn is_ready(&self) -> bool {
        self.entries.read().is_ok()
    }

    /// Get the number of stored patterns.
    pub fn len(&self) -> usize {
        self.entries.read().map(|e| e.len()).unwrap_or(0)
    }

    /// Check if cache is empty.
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Retrieve a cached response using Modern Hopfield attention mechanism.
    ///
    /// Returns `Some(CachedResponse)` if a high-confidence match is found,
    /// `None` for a cache miss.
    ///
    /// # Algorithm
    ///
    /// 1. Compute similarity scores: pattern^T * query (cosine for normalized vectors)
    /// 2. Find the pattern with highest similarity
    /// 3. If raw similarity > MIN_HIT_SIMILARITY, return that pattern's response
    /// 4. Otherwise, return None (cache miss)
    ///
    /// Note: We use raw cosine similarity as the hit threshold, not softmax
    /// attention, because softmax can give 100% attention to a poor match
    /// if it's the only/best option. The raw similarity ensures actual
    /// semantic closeness.
    ///
    /// # Performance
    ///
    /// This MUST complete in <100us for the reflex layer latency budget.
    #[allow(deprecated)]
    pub fn retrieve(&self, query: &[f32]) -> Option<CachedResponse> {
        let start = Instant::now();

        let entries = self.entries.read().ok()?;

        if entries.is_empty() {
            return None;
        }

        // Compute similarity scores (cosine similarity for normalized vectors)
        let mut max_similarity = f32::NEG_INFINITY;
        let mut max_idx = 0;

        // Find best match by raw cosine similarity
        for (idx, entry) in entries.iter().enumerate() {
            let similarity = dot_product_f32(query, &entry.pattern);

            if similarity > max_similarity {
                max_similarity = similarity;
                max_idx = idx;
            }
        }

        // Record retrieval time
        let elapsed_us = start.elapsed().as_micros() as u64;
        self.total_retrieval_us
            .fetch_add(elapsed_us, Ordering::Relaxed);
        self.retrieval_count.fetch_add(1, Ordering::Relaxed);

        // Check if we have a confident hit using raw similarity threshold
        // This ensures actual semantic closeness, not just "best of bad options"
        if max_similarity >= MIN_HIT_SIMILARITY {
            let mut result = entries[max_idx].clone();
            result.access_count += 1;
            result.last_accessed = std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap_or_default()
                .as_millis() as u64;
            Some(result)
        } else {
            None
        }
    }

    /// Store a new pattern-response pair in the cache.
    ///
    /// If the cache is at capacity, removes the least recently accessed entry.
    pub fn store(&self, pattern: &[f32], response: CachedResponse) -> CoreResult<()> {
        let mut entries = self
            .entries
            .write()
            .map_err(|e| CoreError::Internal(format!("Failed to acquire write lock: {}", e)))?;

        // If at capacity, remove least recently accessed
        if entries.len() >= self.capacity {
            if let Some((min_idx, _)) = entries
                .iter()
                .enumerate()
                .min_by_key(|(_, e)| e.last_accessed)
            {
                entries.remove(min_idx);
            }
        }

        // Validate pattern dimension
        if pattern.len() != PATTERN_DIM && !pattern.is_empty() {
            // Allow variable dimensions but log warning in production
            // For now, we accept any dimension > 0
        }

        // Store with the provided pattern
        let mut entry = response;
        entry.pattern = pattern.to_vec();
        entries.push(entry);

        Ok(())
    }

    /// Clear all cached entries.
    pub fn clear(&self) -> CoreResult<()> {
        let mut entries = self
            .entries
            .write()
            .map_err(|e| CoreError::Internal(format!("Failed to acquire write lock: {}", e)))?;
        entries.clear();
        Ok(())
    }

    /// Get cache statistics.
    pub fn stats(&self, hit_count: u64, miss_count: u64) -> CacheStats {
        let retrieval_count = self.retrieval_count.load(Ordering::Relaxed);
        let total_us = self.total_retrieval_us.load(Ordering::Relaxed);

        CacheStats {
            hit_count,
            miss_count,
            patterns_stored: self.len(),
            capacity: self.capacity,
            avg_retrieval_us: if retrieval_count > 0 {
                total_us as f64 / retrieval_count as f64
            } else {
                0.0
            },
            beta: self.beta,
        }
    }
}

impl Default for ModernHopfieldCache {
    fn default() -> Self {
        Self::new()
    }
}
