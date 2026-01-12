//! L2 Reflex Layer implementation.
//!
//! The Reflex layer provides instant pattern-matched responses using a Modern
//! Hopfield Network (MHN) for associative memory lookup. This is the FAST PATH
//! that bypasses deeper processing when high-confidence cached responses exist.

use async_trait::async_trait;
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::{Duration, Instant};

use crate::error::CoreResult;
use crate::traits::NervousLayer;
use crate::types::{LayerId, LayerInput, LayerOutput, LayerResult};

use super::cache::ModernHopfieldCache;
use super::math::{normalize_vector, vector_norm};
use super::types::{CacheStats, CachedResponse, PATTERN_DIM};

// ============================================================
// L2 Reflex Layer
// ============================================================

/// L2 Reflex Layer - Fast path cache lookup using Modern Hopfield Network.
///
/// This layer provides instant pattern-matched responses by looking up
/// cached patterns in the MHN. Cache hits return immediately, cache misses
/// propagate to downstream layers for full processing.
///
/// # Constitution Compliance
///
/// - Latency: <100us (CRITICAL)
/// - Components: MHN cache lookup
/// - UTL: bypass if confidence>0.95
///
/// # No Fallbacks
///
/// Per AP-007: If the Hopfield lookup mechanism fails (not a miss, but a
/// failure), this layer will return an error. Cache misses are NOT errors.
#[derive(Debug)]
pub struct ReflexLayer {
    /// The Modern Hopfield Network cache
    hopfield_cache: ModernHopfieldCache,

    /// Total cache hits
    hit_count: AtomicU64,

    /// Total cache misses
    miss_count: AtomicU64,
}

impl Default for ReflexLayer {
    fn default() -> Self {
        Self::new()
    }
}

impl ReflexLayer {
    /// Create a new Reflex layer with default settings.
    pub fn new() -> Self {
        Self {
            hopfield_cache: ModernHopfieldCache::new(),
            hit_count: AtomicU64::new(0),
            miss_count: AtomicU64::new(0),
        }
    }

    /// Create a Reflex layer with custom cache capacity.
    pub fn with_capacity(capacity: usize) -> Self {
        Self {
            hopfield_cache: ModernHopfieldCache::with_capacity(capacity),
            hit_count: AtomicU64::new(0),
            miss_count: AtomicU64::new(0),
        }
    }

    /// Create a Reflex layer with custom cache settings.
    pub fn with_cache(cache: ModernHopfieldCache) -> Self {
        Self {
            hopfield_cache: cache,
            hit_count: AtomicU64::new(0),
            miss_count: AtomicU64::new(0),
        }
    }

    /// Extract query vector from input.
    ///
    /// Uses pre-computed embedding if available, otherwise computes a
    /// simple hash-based vector for lookup (NOT for production - L1 should
    /// always provide embedding).
    pub(crate) fn get_query_vector(&self, input: &LayerInput) -> CoreResult<Vec<f32>> {
        // Prefer pre-computed embedding from L1
        if let Some(ref embedding) = input.embedding {
            // Truncate or pad to PATTERN_DIM if needed
            let mut query = vec![0.0f32; PATTERN_DIM];
            let copy_len = embedding.len().min(PATTERN_DIM);
            query[..copy_len].copy_from_slice(&embedding[..copy_len]);

            // Normalize
            normalize_vector(&mut query);
            return Ok(query);
        }

        // Fallback: compute simple content-based vector
        // This is NOT ideal - L1 should provide embedding
        // But we need something for cache lookup to work
        let mut query = vec![0.0f32; PATTERN_DIM];

        // Simple hash-based embedding from content
        for (i, byte) in input.content.bytes().enumerate() {
            let idx = i % PATTERN_DIM;
            query[idx] += (byte as f32 - 128.0) / 128.0;
        }

        // Normalize
        normalize_vector(&mut query);

        Ok(query)
    }

    /// Store a new pattern-response pair in the cache.
    ///
    /// Called by L4 Learning after successful processing to cache
    /// responses for future instant retrieval.
    pub fn learn_pattern(&self, pattern: &[f32], response: CachedResponse) -> CoreResult<()> {
        self.hopfield_cache.store(pattern, response)
    }

    /// Get cache statistics.
    pub fn stats(&self) -> CacheStats {
        self.hopfield_cache.stats(
            self.hit_count.load(Ordering::Relaxed),
            self.miss_count.load(Ordering::Relaxed),
        )
    }

    /// Clear the cache.
    pub fn clear_cache(&self) -> CoreResult<()> {
        self.hopfield_cache.clear()
    }

    /// Get the underlying cache for direct access.
    pub fn cache(&self) -> &ModernHopfieldCache {
        &self.hopfield_cache
    }
}

#[async_trait]
impl NervousLayer for ReflexLayer {
    async fn process(&self, input: LayerInput) -> CoreResult<LayerOutput> {
        let start = Instant::now();

        // Extract query vector from input
        let query = self.get_query_vector(&input)?;

        // Try cache lookup
        let (cache_hit, result_data) = match self.hopfield_cache.retrieve(&query) {
            Some(cached) => {
                self.hit_count.fetch_add(1, Ordering::Relaxed);

                // Return cached response (fast path)
                let data = serde_json::json!({
                    "cache_hit": true,
                    "cached_id": cached.id,
                    "confidence": cached.confidence,
                    "response": cached.response_data,
                    "access_count": cached.access_count,
                });
                (true, data)
            }
            None => {
                self.miss_count.fetch_add(1, Ordering::Relaxed);

                // Return miss indicator - downstream layers will process
                let data = serde_json::json!({
                    "cache_hit": false,
                    "query_norm": vector_norm(&query),
                });
                (false, data)
            }
        };

        let duration = start.elapsed();
        let duration_us = duration.as_micros() as u64;

        // Check latency budget
        let budget = self.latency_budget();
        if duration > budget {
            tracing::warn!(
                "ReflexLayer exceeded latency budget: {:?} > {:?}",
                duration,
                budget
            );
        }

        // Build result
        let mut result_json = result_data;
        result_json["duration_us"] = serde_json::json!(duration_us);
        result_json["within_budget"] = serde_json::json!(duration <= budget);

        // Update pulse based on cache hit/miss
        let mut pulse = input.context.pulse.clone();
        if cache_hit {
            // Cache hit reduces entropy (known pattern)
            pulse.entropy = (pulse.entropy * 0.5).max(0.0);
        }

        Ok(LayerOutput {
            layer: LayerId::Reflex,
            result: LayerResult::success(LayerId::Reflex, result_json),
            pulse,
            duration_us,
        })
    }

    fn latency_budget(&self) -> Duration {
        Duration::from_micros(100) // 100us - CRITICAL!
    }

    fn layer_id(&self) -> LayerId {
        LayerId::Reflex
    }

    fn layer_name(&self) -> &'static str {
        "Reflex Layer"
    }

    async fn health_check(&self) -> CoreResult<bool> {
        // Check cache is initialized and responsive
        if !self.hopfield_cache.is_ready() {
            return Ok(false);
        }

        // Quick retrieval test with empty query
        let test_query = vec![0.0f32; PATTERN_DIM];
        let start = Instant::now();
        let _ = self.hopfield_cache.retrieve(&test_query);
        let elapsed = start.elapsed();

        // Should complete well within budget even for empty cache
        Ok(elapsed < Duration::from_micros(500))
    }
}
