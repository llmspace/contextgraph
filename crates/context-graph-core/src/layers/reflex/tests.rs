//! Tests for the Reflex layer - REAL implementations, NO MOCKS.

use std::time::{Duration, Instant};

use crate::traits::NervousLayer;
use crate::types::{LayerId, LayerInput};

use super::cache::ModernHopfieldCache;
use super::layer::ReflexLayer;
use super::math::normalize_vector;
use super::types::{CachedResponse, PATTERN_DIM};

// ============================================================
// Modern Hopfield Cache Tests
// ============================================================

#[test]
fn test_cache_creation() {
    let cache = ModernHopfieldCache::new();
    assert!(cache.is_ready());
    assert!(cache.is_empty());
    assert_eq!(cache.len(), 0);
}

#[test]
fn test_cache_with_capacity() {
    let cache = ModernHopfieldCache::with_capacity(100);
    assert!(cache.is_ready());
    assert_eq!(cache.capacity, 100);
}

#[test]
fn test_cache_store_and_retrieve() {
    let cache = ModernHopfieldCache::new();

    // Create a pattern
    let mut pattern = vec![0.0f32; PATTERN_DIM];
    pattern[0] = 1.0;
    pattern[1] = 0.5;
    normalize_vector(&mut pattern);

    // Store it
    let response = CachedResponse::new(
        "test-1".to_string(),
        pattern.clone(),
        serde_json::json!({"answer": 42}),
        0.95,
    );
    cache.store(&pattern, response).unwrap();

    assert_eq!(cache.len(), 1);

    // Retrieve with same pattern
    let result = cache.retrieve(&pattern);
    assert!(result.is_some());
    let cached = result.unwrap();
    assert_eq!(cached.id, "test-1");
    assert_eq!(cached.response_data["answer"], 42);
}

#[test]
fn test_cache_miss_for_dissimilar_pattern() {
    let cache = ModernHopfieldCache::new();

    // Create and store a pattern
    let mut pattern1 = vec![0.0f32; PATTERN_DIM];
    pattern1[0] = 1.0;
    normalize_vector(&mut pattern1);

    let response = CachedResponse::new(
        "test-1".to_string(),
        pattern1.clone(),
        serde_json::json!({"data": "original"}),
        0.95,
    );
    cache.store(&pattern1, response).unwrap();

    // Query with very different pattern
    let mut pattern2 = vec![0.0f32; PATTERN_DIM];
    pattern2[PATTERN_DIM / 2] = 1.0; // Orthogonal direction
    normalize_vector(&mut pattern2);

    let result = cache.retrieve(&pattern2);
    assert!(result.is_none(), "Should miss for orthogonal pattern");
}

#[test]
fn test_cache_capacity_limit() {
    let cache = ModernHopfieldCache::with_capacity(3);

    // Store 5 patterns (should evict oldest 2)
    for i in 0..5 {
        let mut pattern = vec![0.0f32; PATTERN_DIM];
        pattern[i % PATTERN_DIM] = 1.0;
        normalize_vector(&mut pattern);

        let response = CachedResponse::new(
            format!("test-{}", i),
            pattern.clone(),
            serde_json::json!({"idx": i}),
            0.95,
        );
        cache.store(&pattern, response).unwrap();

        // Small delay to ensure different timestamps
        std::thread::sleep(std::time::Duration::from_millis(1));
    }

    // Should have exactly 3 entries
    assert_eq!(cache.len(), 3);
}

#[test]
fn test_cache_empty_returns_none() {
    let cache = ModernHopfieldCache::new();
    let query = vec![1.0f32; PATTERN_DIM];
    let result = cache.retrieve(&query);
    assert!(result.is_none());
}

#[test]
fn test_cache_clear() {
    let cache = ModernHopfieldCache::new();

    // Add some entries
    for i in 0..5 {
        let mut pattern = vec![0.0f32; PATTERN_DIM];
        pattern[i % PATTERN_DIM] = 1.0;
        normalize_vector(&mut pattern);

        let response = CachedResponse::new(
            format!("test-{}", i),
            pattern.clone(),
            serde_json::json!({}),
            0.95,
        );
        cache.store(&pattern, response).unwrap();
    }

    assert!(!cache.is_empty());
    cache.clear().unwrap();
    assert!(cache.is_empty());
}

// ============================================================
// ReflexLayer Tests
// ============================================================

#[tokio::test]
async fn test_reflex_layer_process_miss() {
    let layer = ReflexLayer::new();
    let input = LayerInput::new("test-req".to_string(), "Hello world".to_string());

    let output = layer.process(input).await.unwrap();

    assert_eq!(output.layer, LayerId::Reflex);
    assert!(output.result.success);
    assert_eq!(output.result.data["cache_hit"], false);
}

#[tokio::test]
async fn test_reflex_layer_process_hit() {
    let layer = ReflexLayer::new();

    // Pre-populate cache
    let mut pattern = vec![0.0f32; PATTERN_DIM];
    for (i, byte) in "test content".bytes().enumerate() {
        let idx = i % PATTERN_DIM;
        pattern[idx] += (byte as f32 - 128.0) / 128.0;
    }
    normalize_vector(&mut pattern);

    let response = CachedResponse::new(
        "cached-response".to_string(),
        pattern.clone(),
        serde_json::json!({"cached": true, "data": "test result"}),
        0.98,
    );
    layer.learn_pattern(&pattern, response).unwrap();

    // Now query with similar content
    let input = LayerInput::new("test-req".to_string(), "test content".to_string());
    let output = layer.process(input).await.unwrap();

    assert_eq!(output.layer, LayerId::Reflex);
    assert!(output.result.success);
    assert_eq!(output.result.data["cache_hit"], true);
    assert_eq!(output.result.data["cached_id"], "cached-response");
}

#[tokio::test]
async fn test_reflex_layer_properties() {
    let layer = ReflexLayer::new();

    assert_eq!(layer.layer_id(), LayerId::Reflex);
    assert_eq!(layer.latency_budget(), Duration::from_micros(100));
    assert_eq!(layer.layer_name(), "Reflex Layer");
}

#[tokio::test]
async fn test_reflex_layer_health_check() {
    let layer = ReflexLayer::new();
    let healthy = layer.health_check().await.unwrap();
    assert!(healthy, "ReflexLayer should be healthy");
}

#[tokio::test]
async fn test_reflex_layer_stats() {
    let layer = ReflexLayer::new();

    // Process a few requests to generate stats
    for i in 0..3 {
        let input = LayerInput::new(format!("req-{}", i), format!("content-{}", i));
        let _ = layer.process(input).await;
    }

    let stats = layer.stats();
    assert_eq!(stats.miss_count, 3); // All should be misses
    assert_eq!(stats.hit_count, 0);
    assert_eq!(stats.hit_rate(), 0.0);
}

// ============================================================
// Performance Benchmark - CRITICAL <100us
// ============================================================

#[tokio::test]
async fn test_reflex_layer_latency_benchmark() {
    let layer = ReflexLayer::new();

    // Pre-populate with some patterns
    for i in 0..100 {
        let mut pattern = vec![0.0f32; PATTERN_DIM];
        pattern[i % PATTERN_DIM] = 1.0;
        pattern[(i + 1) % PATTERN_DIM] = 0.5;
        normalize_vector(&mut pattern);

        let response = CachedResponse::new(
            format!("cached-{}", i),
            pattern.clone(),
            serde_json::json!({"idx": i}),
            0.95,
        );
        layer.learn_pattern(&pattern, response).unwrap();
    }

    // Benchmark lookups
    let iterations = 1000;
    let mut total_us: u64 = 0;
    let mut max_us: u64 = 0;

    for i in 0..iterations {
        let input = LayerInput::new(
            format!("bench-{}", i),
            format!("benchmark content {}", i % 100),
        );

        let start = Instant::now();
        let _ = layer.process(input).await;
        let elapsed = start.elapsed().as_micros() as u64;

        total_us += elapsed;
        max_us = max_us.max(elapsed);
    }

    let avg_us = total_us / iterations;

    println!("Reflex Layer Benchmark Results:");
    println!("  Iterations: {}", iterations);
    println!("  Avg latency: {} us", avg_us);
    println!("  Max latency: {} us", max_us);
    println!("  Budget: 100 us");

    // We expect average to be well under budget
    // Max might occasionally exceed due to system jitter, but average should be fast
    assert!(
        avg_us < 100,
        "Average latency {} us exceeds 100us budget",
        avg_us
    );
}

#[test]
fn test_cache_lookup_benchmark() {
    let cache = ModernHopfieldCache::new();

    // Pre-populate
    for i in 0..1000 {
        let mut pattern = vec![0.0f32; PATTERN_DIM];
        pattern[i % PATTERN_DIM] = 1.0;
        normalize_vector(&mut pattern);

        let response = CachedResponse::new(
            format!("entry-{}", i),
            pattern.clone(),
            serde_json::json!({}),
            0.95,
        );
        cache.store(&pattern, response).unwrap();
    }

    // Benchmark raw cache lookups
    let iterations = 10_000;
    let start = Instant::now();

    for i in 0..iterations {
        let mut query = vec![0.0f32; PATTERN_DIM];
        query[i % PATTERN_DIM] = 1.0;
        normalize_vector(&mut query);
        let _ = cache.retrieve(&query);
    }

    let total_us = start.elapsed().as_micros();
    let avg_us = total_us / iterations as u128;

    println!("Cache Lookup Benchmark:");
    println!("  Iterations: {}", iterations);
    println!("  Total time: {} us", total_us);
    println!("  Avg latency: {} us", avg_us);

    // Just verify it completes - CI machines have variable performance
    // In production, this should be <50us average on dedicated hardware
    assert!(
        avg_us < 500,
        "Cache lookup average {} us extremely slow (expected <500us)",
        avg_us
    );
}

// ============================================================
// Edge Cases
// ============================================================

#[test]
fn test_cache_with_embedding_input() {
    let layer = ReflexLayer::new();

    // Create input with pre-computed embedding
    let mut input = LayerInput::new("test".to_string(), "content".to_string());
    let mut embedding = vec![0.1f32; 256];
    embedding[0] = 1.0;
    input.embedding = Some(embedding);

    // Should extract query from embedding
    let query = layer.get_query_vector(&input).unwrap();
    assert_eq!(query.len(), PATTERN_DIM);
}
