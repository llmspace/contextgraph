//! Cache key type for the embedding cache system.
//!
//! This module provides [`CacheKey`], a unique key derived from xxHash64 content hash.

use serde::{Deserialize, Serialize};
use xxhash_rust::xxh64::xxh64;

use crate::types::{FusedEmbedding, ModelInput};

/// Cache key derived from xxHash64 content hash.
///
/// # Design Rationale
/// - `Copy` + `Eq` + `Hash` enables direct HashMap key usage
/// - 8 bytes = single register, no allocation
/// - xxHash64 collision probability: ~1/2^64
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct CacheKey {
    /// xxHash64 of content (from ModelInput::content_hash() or FusedEmbedding.content_hash)
    pub content_hash: u64,
}

impl CacheKey {
    /// Create key from raw text content.
    /// Uses xxHash64 internally (same as ModelInput::content_hash).
    #[must_use]
    pub fn from_content(content: &str) -> Self {
        Self {
            content_hash: xxh64(content.as_bytes(), 0),
        }
    }

    /// Create key from ModelInput.
    /// Simply wraps ModelInput::content_hash() which already uses xxHash64.
    #[must_use]
    pub fn from_input(input: &ModelInput) -> Self {
        Self {
            content_hash: input.content_hash(),
        }
    }

    /// Create key from FusedEmbedding.
    /// Uses the pre-computed content_hash field.
    #[must_use]
    pub fn from_embedding(embedding: &FusedEmbedding) -> Self {
        Self {
            content_hash: embedding.content_hash,
        }
    }
}

impl From<u64> for CacheKey {
    fn from(hash: u64) -> Self {
        Self { content_hash: hash }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::dimensions::FUSED_OUTPUT;

    fn make_real_fused_embedding() -> FusedEmbedding {
        let vector = vec![0.1f32; FUSED_OUTPUT];
        let weights = [0.125f32; 8];
        FusedEmbedding::new(vector, weights, [0, 1, 2, 3], 100, 0xDEADBEEF)
            .expect("Test helper should create valid embedding")
    }

    #[test]
    fn test_cache_key_from_content_same_content_same_hash() {
        println!("BEFORE: Creating CacheKey from 'Hello, World!'");
        let key1 = CacheKey::from_content("Hello, World!");
        let key2 = CacheKey::from_content("Hello, World!");

        println!("AFTER: key1.content_hash = {:#x}", key1.content_hash);
        println!("AFTER: key2.content_hash = {:#x}", key2.content_hash);

        assert_eq!(key1, key2);
        assert_eq!(key1.content_hash, key2.content_hash);
        println!("PASSED: Same content produces identical hash");
    }

    #[test]
    fn test_cache_key_from_content_different_content_different_hash() {
        println!("BEFORE: Creating CacheKeys from different content");
        let key1 = CacheKey::from_content("Hello");
        let key2 = CacheKey::from_content("World");

        println!("AFTER: key1.content_hash = {:#x}", key1.content_hash);
        println!("AFTER: key2.content_hash = {:#x}", key2.content_hash);

        assert_ne!(key1, key2);
        assert_ne!(key1.content_hash, key2.content_hash);
        println!("PASSED: Different content produces different hash");
    }

    #[test]
    fn test_cache_key_from_input_matches_model_input_content_hash() {
        let input = ModelInput::text("Test content for hashing").unwrap();
        let expected_hash = input.content_hash();

        println!("BEFORE: ModelInput::content_hash() = {:#x}", expected_hash);

        let key = CacheKey::from_input(&input);

        println!("AFTER: CacheKey.content_hash = {:#x}", key.content_hash);

        assert_eq!(key.content_hash, expected_hash);
        println!("PASSED: CacheKey::from_input() matches ModelInput::content_hash()");
    }

    #[test]
    fn test_cache_key_from_embedding_matches_fused_embedding_content_hash() {
        let embedding = make_real_fused_embedding();
        let expected_hash = embedding.content_hash;

        println!("BEFORE: FusedEmbedding.content_hash = {:#x}", expected_hash);

        let key = CacheKey::from_embedding(&embedding);

        println!("AFTER: CacheKey.content_hash = {:#x}", key.content_hash);

        assert_eq!(key.content_hash, expected_hash);
        println!("PASSED: CacheKey::from_embedding() matches FusedEmbedding.content_hash");
    }

    #[test]
    fn test_cache_key_from_u64() {
        let hash = 0x123456789ABCDEF0_u64;

        println!("BEFORE: Raw hash = {:#x}", hash);

        let key: CacheKey = hash.into();

        println!("AFTER: CacheKey.content_hash = {:#x}", key.content_hash);

        assert_eq!(key.content_hash, hash);
        println!("PASSED: CacheKey::from(u64) works correctly");
    }

    #[test]
    fn test_cache_key_is_copy() {
        let key = CacheKey::from_content("copy test");
        let key_copy = key; // This should work since CacheKey is Copy

        assert_eq!(key, key_copy);
        println!("PASSED: CacheKey is Copy (8 bytes, no allocation)");
    }

    #[test]
    fn test_cache_key_size_is_8_bytes() {
        let size = std::mem::size_of::<CacheKey>();

        println!("CacheKey size = {} bytes", size);

        assert_eq!(size, 8);
        println!("PASSED: CacheKey is exactly 8 bytes");
    }

    #[test]
    fn test_edge_case_empty_string_cache_key() {
        println!("BEFORE: Creating CacheKey from empty string");

        let key = CacheKey::from_content("");

        println!("AFTER: CacheKey.content_hash = {:#x}", key.content_hash);

        // xxHash64 of empty string is NOT zero
        assert_ne!(
            key.content_hash, 0,
            "Empty string should produce non-zero hash"
        );
        println!("PASSED: Empty string produces valid non-zero hash");
    }

    #[test]
    fn test_edge_case_cache_key_hash_properties() {
        // Test that CacheKey implements Hash correctly
        use std::collections::HashMap;

        let key1 = CacheKey::from_content("test1");
        let key2 = CacheKey::from_content("test2");
        let key1_dup = CacheKey::from_content("test1");

        let mut map: HashMap<CacheKey, &str> = HashMap::new();
        map.insert(key1, "value1");
        map.insert(key2, "value2");

        println!("BEFORE: HashMap with 2 keys");
        println!("AFTER: map.get(key1) = {:?}", map.get(&key1));
        println!("AFTER: map.get(key1_dup) = {:?}", map.get(&key1_dup));
        println!("AFTER: map.get(key2) = {:?}", map.get(&key2));

        assert_eq!(map.get(&key1), Some(&"value1"));
        assert_eq!(map.get(&key1_dup), Some(&"value1")); // Same content = same key
        assert_eq!(map.get(&key2), Some(&"value2"));

        println!("PASSED: CacheKey works correctly as HashMap key");
    }
}
