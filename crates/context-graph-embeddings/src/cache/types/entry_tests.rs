//! Tests for CacheEntry type.

use super::*;
use crate::types::dimensions::FUSED_OUTPUT;

fn make_real_fused_embedding() -> FusedEmbedding {
    let vector = vec![0.1f32; FUSED_OUTPUT];
    let weights = [0.125f32; 8];
    FusedEmbedding::new(vector, weights, [0, 1, 2, 3], 100, 0xDEADBEEF)
        .expect("Test helper should create valid embedding")
}

#[test]
fn test_cache_entry_new_sets_access_count_to_1() {
    println!("BEFORE: Creating CacheEntry");

    let entry = CacheEntry::new(make_real_fused_embedding());

    println!("AFTER: access_count = {}", entry.access_count());

    assert_eq!(entry.access_count(), 1);
    println!("PASSED: CacheEntry::new() sets access_count to 1");
}

#[test]
fn test_cache_entry_touch_updates_last_accessed() {
    let entry = CacheEntry::new(make_real_fused_embedding());
    let initial = entry.last_accessed();

    println!("BEFORE: last_accessed = {:?}", initial);

    std::thread::sleep(std::time::Duration::from_millis(10));
    entry.touch();

    let after = entry.last_accessed();

    println!("AFTER: last_accessed = {:?}", after);

    assert!(
        after > initial,
        "last_accessed should increase after touch()"
    );
    println!("PASSED: CacheEntry::touch() updates last_accessed");
}

#[test]
fn test_cache_entry_increment_access_increases_count() {
    let entry = CacheEntry::new(make_real_fused_embedding());

    println!("BEFORE: access_count = {}", entry.access_count());

    entry.increment_access();
    entry.increment_access();
    entry.increment_access();

    println!("AFTER: access_count = {}", entry.access_count());

    assert_eq!(entry.access_count(), 4); // 1 initial + 3 increments
    println!("PASSED: CacheEntry::increment_access() increases count");
}

#[test]
fn test_cache_entry_is_expired_with_zero_ttl() {
    let entry = CacheEntry::new(make_real_fused_embedding());

    println!("BEFORE: age = {:?}", entry.age());
    println!(
        "BEFORE: is_expired(0s) = {}",
        entry.is_expired(Duration::ZERO)
    );

    // With TTL of 0, entry should be expired immediately
    assert!(
        entry.is_expired(Duration::ZERO),
        "Zero TTL should expire immediately"
    );

    // With large TTL, entry should NOT be expired
    assert!(
        !entry.is_expired(Duration::from_secs(3600)),
        "1 hour TTL should not expire"
    );

    println!("AFTER: Expiration logic verified");
    println!("PASSED: Zero TTL expires immediately, large TTL does not");
}

#[test]
fn test_cache_entry_is_expired_after_ttl() {
    let entry = CacheEntry::new(make_real_fused_embedding());

    println!("BEFORE: age = {:?}", entry.age());

    // Wait a bit
    std::thread::sleep(std::time::Duration::from_millis(50));

    let age = entry.age();
    println!("AFTER: age = {:?}", age);

    // Should be expired with 10ms TTL
    assert!(
        entry.is_expired(Duration::from_millis(10)),
        "Should be expired after 50ms with 10ms TTL"
    );

    // Should NOT be expired with 1s TTL
    assert!(
        !entry.is_expired(Duration::from_secs(1)),
        "Should not be expired with 1s TTL"
    );

    println!("PASSED: is_expired() correctly checks against TTL");
}

#[test]
fn test_cache_entry_memory_size_at_least_6200_bytes() {
    let entry = CacheEntry::new(make_real_fused_embedding());

    let size = entry.memory_size();

    println!("BEFORE: Creating CacheEntry");
    println!("AFTER: CacheEntry.memory_size() = {} bytes", size);

    // FusedEmbedding base: 6198 bytes
    // + Instant: 16 bytes
    // + AtomicU64: 8 bytes
    // + AtomicU32: 4 bytes
    // Total: ~6226 bytes minimum
    assert!(
        size >= 6200,
        "Memory size should be at least 6200 bytes, got {}",
        size
    );
    println!("PASSED: CacheEntry.memory_size() >= 6200 bytes");
}

#[test]
fn test_cache_entry_created_at_returns_instant() {
    let before = Instant::now();
    let entry = CacheEntry::new(make_real_fused_embedding());
    let after = Instant::now();

    let created_at = entry.created_at();

    println!("BEFORE: Instant::now() = {:?}", before);
    println!("AFTER: entry.created_at() = {:?}", created_at);

    assert!(created_at >= before);
    assert!(created_at <= after);
    println!("PASSED: created_at() returns valid Instant");
}

#[test]
fn test_edge_case_access_count_saturation() {
    let entry = CacheEntry::new(make_real_fused_embedding());

    println!("BEFORE: access_count = {}", entry.access_count());

    // Simulate many accesses
    for _ in 0..1000 {
        entry.increment_access();
    }

    println!("AFTER: access_count = {}", entry.access_count());

    assert_eq!(
        entry.access_count(),
        1001,
        "Count should be 1 (initial) + 1000"
    );
    println!("PASSED: Access count increments correctly");
}

#[test]
fn test_edge_case_zero_ttl_expires_immediately() {
    let entry = CacheEntry::new(make_real_fused_embedding());

    println!("BEFORE: age = {:?}", entry.age());
    println!(
        "BEFORE: is_expired(0s) = {}",
        entry.is_expired(Duration::ZERO)
    );

    // With TTL of 0, entry should be expired immediately
    assert!(
        entry.is_expired(Duration::ZERO),
        "Zero TTL should expire immediately"
    );

    // With large TTL, entry should NOT be expired
    assert!(
        !entry.is_expired(Duration::from_secs(3600)),
        "1 hour TTL should not expire"
    );

    println!("AFTER: Expiration logic verified");
    println!("PASSED: Zero TTL expires immediately, large TTL does not");
}

#[test]
fn test_cache_entry_embedding_is_accessible() {
    let original_hash = 0xDEADBEEF_u64;
    let entry = CacheEntry::new(make_real_fused_embedding());

    println!(
        "BEFORE: entry.embedding.content_hash = {:#x}",
        original_hash
    );
    println!(
        "AFTER: entry.embedding.content_hash = {:#x}",
        entry.embedding.content_hash
    );

    assert_eq!(entry.embedding.content_hash, original_hash);
    assert_eq!(entry.embedding.vector.len(), FUSED_OUTPUT);
    println!("PASSED: CacheEntry.embedding is publicly accessible");
}
