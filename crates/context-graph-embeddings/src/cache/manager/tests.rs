//! Tests for CacheManager core functionality.

use std::sync::atomic::Ordering;

use crate::cache::types::CacheKey;
use crate::config::{CacheConfig, EvictionPolicy};
use crate::types::dimensions::FUSED_OUTPUT;
use crate::types::FusedEmbedding;

use super::core::CacheManager;

// ========== Test Helpers ==========

fn create_test_embedding(content_hash: u64) -> FusedEmbedding {
    let vector: Vec<f32> = (0..FUSED_OUTPUT)
        .map(|i| ((i as f32 + content_hash as f32) % 2.0) - 1.0)
        .collect();
    let weights = [0.125f32; 8];
    FusedEmbedding::new(vector, weights, [0, 1, 2, 3], 100, content_hash)
        .expect("Test helper should create valid embedding")
}

fn create_test_config() -> CacheConfig {
    CacheConfig {
        enabled: true,
        max_entries: 100,
        max_bytes: 1_000_000,
        ttl_seconds: None,
        eviction_policy: EvictionPolicy::Lru,
        persist_to_disk: false,
        disk_path: None,
    }
}

// ========== CacheManager::new() Tests ==========

#[test]
fn test_new_with_valid_config() {
    let config = create_test_config();
    let result = CacheManager::new(config);
    assert!(result.is_ok());
    let cache = result.unwrap();
    assert!(cache.is_empty());
    assert_eq!(cache.len(), 0);
}

#[test]
fn test_new_with_zero_max_entries_fails() {
    let config = CacheConfig {
        max_entries: 0,
        ..create_test_config()
    };
    assert!(CacheManager::new(config).is_err());
}

#[test]
fn test_new_with_zero_max_bytes_fails() {
    let config = CacheConfig {
        max_bytes: 0,
        ..create_test_config()
    };
    assert!(CacheManager::new(config).is_err());
}

#[test]
fn test_new_with_persist_but_no_path_fails() {
    let config = CacheConfig {
        persist_to_disk: true,
        disk_path: None,
        ..create_test_config()
    };
    assert!(CacheManager::new(config).is_err());
}

// ========== put/get Round-Trip Tests ==========

#[test]
fn test_put_get_roundtrip() {
    let config = create_test_config();
    let cache = CacheManager::new(config).unwrap();

    let key = CacheKey::from_content("test content");
    let embedding = create_test_embedding(12345);
    let original_hash = embedding.content_hash;

    cache.put(key, embedding).unwrap();
    let retrieved = cache.get(&key);

    assert!(retrieved.is_some());
    assert_eq!(retrieved.unwrap().content_hash, original_hash);
}

#[test]
fn test_get_nonexistent_key_returns_none() {
    let config = create_test_config();
    let cache = CacheManager::new(config).unwrap();

    let key = CacheKey::from_content("nonexistent");
    assert!(cache.get(&key).is_none());
    assert_eq!(cache.metrics.misses.load(Ordering::Relaxed), 1);
}

// ========== LRU Eviction Tests ==========

#[test]
fn test_lru_eviction_at_max_entries() {
    let config = CacheConfig {
        max_entries: 3,
        max_bytes: 100_000_000,
        ..create_test_config()
    };
    let cache = CacheManager::new(config).unwrap();

    let key1 = CacheKey::from(1u64);
    let key2 = CacheKey::from(2u64);
    let key3 = CacheKey::from(3u64);

    cache.put(key1, create_test_embedding(1)).unwrap();
    cache.put(key2, create_test_embedding(2)).unwrap();
    cache.put(key3, create_test_embedding(3)).unwrap();

    assert_eq!(cache.len(), 3);

    let key4 = CacheKey::from(4u64);
    cache.put(key4, create_test_embedding(4)).unwrap();

    assert_eq!(cache.len(), 3);
    assert!(!cache.contains(&key1), "key1 should have been evicted");
    assert!(cache.contains(&key4));
    assert_eq!(cache.metrics.evictions.load(Ordering::Relaxed), 1);
}

#[test]
fn test_lru_access_prevents_eviction() {
    let config = CacheConfig {
        max_entries: 3,
        max_bytes: 100_000_000,
        ..create_test_config()
    };
    let cache = CacheManager::new(config).unwrap();

    let key_a = CacheKey::from(0xA_u64);
    let key_b = CacheKey::from(0xB_u64);
    let key_c = CacheKey::from(0xC_u64);

    cache.put(key_a, create_test_embedding(0xA)).unwrap();
    cache.put(key_b, create_test_embedding(0xB)).unwrap();
    cache.put(key_c, create_test_embedding(0xC)).unwrap();

    // Access A - moves it to back of LRU order
    assert!(cache.get(&key_a).is_some());

    // Insert D - triggers eviction of B (oldest accessed)
    let key_d = CacheKey::from(0xD_u64);
    cache.put(key_d, create_test_embedding(0xD)).unwrap();

    assert!(cache.contains(&key_a), "A should NOT be evicted");
    assert!(!cache.contains(&key_b), "B SHOULD be evicted");
}

#[test]
fn test_max_entries_limit_enforced() {
    let config = CacheConfig {
        max_entries: 5,
        max_bytes: 100_000_000,
        ..create_test_config()
    };
    let cache = CacheManager::new(config).unwrap();

    for i in 0..10 {
        cache.put(CacheKey::from(i as u64), create_test_embedding(i)).unwrap();
    }

    assert_eq!(cache.len(), 5);
    assert_eq!(cache.metrics.evictions.load(Ordering::Relaxed), 5);
}

#[test]
fn test_max_bytes_limit_enforced() {
    let config = CacheConfig {
        max_entries: 100,
        max_bytes: 15_000,
        ..create_test_config()
    };
    let cache = CacheManager::new(config).unwrap();

    for i in 0..5 {
        cache.put(CacheKey::from(i as u64), create_test_embedding(i)).unwrap();
    }

    assert!(cache.memory_usage() <= 15_000);
    assert!(cache.metrics.evictions.load(Ordering::Relaxed) > 0);
}

// ========== contains/remove/clear Tests ==========

#[test]
fn test_contains_key() {
    let config = create_test_config();
    let cache = CacheManager::new(config).unwrap();

    let key = CacheKey::from_content("contains test");
    assert!(!cache.contains(&key));

    cache.put(key, create_test_embedding(222)).unwrap();
    assert!(cache.contains(&key));
}

#[test]
fn test_remove_entry() {
    let config = create_test_config();
    let cache = CacheManager::new(config).unwrap();

    let key = CacheKey::from_content("remove test");
    cache.put(key, create_test_embedding(333)).unwrap();

    assert!(cache.remove(&key).is_some());
    assert_eq!(cache.len(), 0);
    assert!(!cache.contains(&key));
}

#[test]
fn test_clear() {
    let config = create_test_config();
    let cache = CacheManager::new(config).unwrap();

    for i in 0..5 {
        cache.put(CacheKey::from(i as u64), create_test_embedding(i)).unwrap();
    }

    cache.clear();

    assert!(cache.is_empty());
    assert_eq!(cache.memory_usage(), 0);
}
