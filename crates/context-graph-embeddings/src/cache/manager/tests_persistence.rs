//! Tests for CacheManager persistence and edge cases.

use std::path::PathBuf;
use std::sync::atomic::Ordering;
use tempfile::TempDir;

use crate::cache::types::{CacheEntry, CacheKey};
use crate::config::{CacheConfig, EvictionPolicy};
use crate::types::dimensions::FUSED_OUTPUT;
use crate::types::FusedEmbedding;

use super::core::CacheManager;
use super::persistence::{CACHE_MAGIC, CACHE_VERSION};

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

fn create_test_config_with_ttl(ttl_secs: u64) -> CacheConfig {
    CacheConfig {
        ttl_seconds: Some(ttl_secs),
        ..create_test_config()
    }
}

fn create_test_config_with_persistence(path: PathBuf) -> CacheConfig {
    CacheConfig {
        persist_to_disk: true,
        disk_path: Some(path),
        ..create_test_config()
    }
}

// ========== TTL Tests ==========

#[test]
fn test_ttl_expiration_returns_none() {
    let config = create_test_config_with_ttl(0);
    let cache = CacheManager::new(config).unwrap();

    let key = CacheKey::from_content("ttl test");
    cache.put(key, create_test_embedding(999)).unwrap();

    std::thread::sleep(std::time::Duration::from_millis(1));
    assert!(cache.get(&key).is_none());
    assert_eq!(cache.metrics.misses.load(Ordering::Relaxed), 1);
}

#[test]
fn test_ttl_valid_entry_returned() {
    let config = create_test_config_with_ttl(3600);
    let cache = CacheManager::new(config).unwrap();

    let key = CacheKey::from_content("valid ttl");
    cache.put(key, create_test_embedding(888)).unwrap();

    assert!(cache.get(&key).is_some());
    assert_eq!(cache.metrics.hits.load(Ordering::Relaxed), 1);
}

// ========== Hit Rate Tests ==========

#[test]
fn test_hit_rate_calculation() {
    let config = create_test_config();
    let cache = CacheManager::new(config).unwrap();

    assert_eq!(cache.hit_rate(), 0.0);

    let key = CacheKey::from_content("hit rate test");
    cache.put(key, create_test_embedding(111)).unwrap();

    let _ = cache.get(&key);
    assert_eq!(cache.hit_rate(), 1.0);

    let _ = cache.get(&CacheKey::from_content("nonexistent"));
    assert!((cache.hit_rate() - 0.5).abs() < 0.001);
}

// ========== Persistence Tests ==========

#[tokio::test]
async fn test_persist_load_roundtrip() {
    let temp_dir = TempDir::new().unwrap();
    let cache_path = temp_dir.path().join("cache.bin");

    let config = create_test_config_with_persistence(cache_path.clone());
    let cache = CacheManager::new(config).unwrap();

    let key1 = CacheKey::from(100u64);
    let key2 = CacheKey::from(200u64);
    cache.put(key1, create_test_embedding(100)).unwrap();
    cache.put(key2, create_test_embedding(200)).unwrap();

    cache.persist().await.unwrap();
    assert!(cache_path.exists());

    cache.clear();
    assert_eq!(cache.len(), 0);

    cache.load().await.unwrap();

    assert_eq!(cache.len(), 2);
    assert!(cache.get(&key1).is_some());
    assert!(cache.get(&key2).is_some());
}

#[tokio::test]
async fn test_persist_without_path_fails() {
    let config = create_test_config();
    let cache = CacheManager::new(config).unwrap();
    assert!(cache.persist().await.is_err());
}

#[tokio::test]
async fn test_load_detects_checksum_mismatch() {
    let temp_dir = TempDir::new().unwrap();
    let cache_path = temp_dir.path().join("corrupt.bin");

    let mut data = Vec::new();
    data.extend_from_slice(&CACHE_MAGIC);
    data.push(CACHE_VERSION);
    data.extend_from_slice(&0u64.to_le_bytes());
    data.extend_from_slice(&0u64.to_le_bytes()); // wrong checksum
    tokio::fs::write(&cache_path, &data).await.unwrap();

    let config = create_test_config_with_persistence(cache_path);
    let cache = CacheManager::new(config).unwrap();

    assert!(cache.load().await.is_err());
}

// ========== Edge Case Tests ==========

#[test]
fn test_put_oversized_entry_fails() {
    let config = CacheConfig {
        max_entries: 100,
        max_bytes: 100,
        ..create_test_config()
    };
    let cache = CacheManager::new(config).unwrap();

    let key = CacheKey::from_content("oversized");
    assert!(cache.put(key, create_test_embedding(999)).is_err());
}

#[test]
fn test_update_existing_key() {
    let config = create_test_config();
    let cache = CacheManager::new(config).unwrap();

    let key = CacheKey::from_content("update test");

    cache.put(key, create_test_embedding(1)).unwrap();
    let initial_bytes = cache.memory_usage();

    cache.put(key, create_test_embedding(2)).unwrap();

    assert_eq!(cache.len(), 1);
    assert!(cache.memory_usage() <= initial_bytes + 100);
}

#[test]
fn test_memory_tracking_accuracy() {
    let config = create_test_config();
    let cache = CacheManager::new(config).unwrap();

    let key = CacheKey::from_content("memory test");
    let expected_size = CacheEntry::new(create_test_embedding(0)).memory_size();

    cache.put(key, create_test_embedding(555)).unwrap();
    assert_eq!(cache.memory_usage(), expected_size);

    cache.remove(&key);
    assert_eq!(cache.memory_usage(), 0);
}
