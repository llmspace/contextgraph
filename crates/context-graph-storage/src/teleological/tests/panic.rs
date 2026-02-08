//! Panic behavior tests (should_panic).

use crate::teleological::*;
use uuid::Uuid;

// =========================================================================
// PANIC TESTS (Verify fail-fast behavior)
// =========================================================================

#[test]
#[should_panic(expected = "DESERIALIZATION ERROR")]
fn test_panic_on_empty_fingerprint_data() {
    let _ = deserialize_teleological_fingerprint(&[]);
}

#[test]
#[should_panic(expected = "DESERIALIZATION ERROR")]
fn test_panic_on_wrong_version() {
    let mut data = vec![255u8]; // Wrong version
    data.extend(vec![0u8; 100]); // Garbage data
    let _ = deserialize_teleological_fingerprint(&data);
}

#[test]
#[should_panic(expected = "DESERIALIZATION ERROR")]
fn test_panic_on_wrong_topic_profile_size() {
    let _ = deserialize_topic_profile(&[0u8; 51]); // Should be 52
}

#[test]
#[should_panic(expected = "DESERIALIZATION ERROR")]
fn test_panic_on_wrong_e1_vector_size() {
    let _ = deserialize_e1_matryoshka_128(&[0u8; 500]); // Should be 512
}

#[test]
#[should_panic(expected = "DESERIALIZATION ERROR")]
fn test_panic_on_truncated_memory_id_list() {
    // Create valid list, then truncate
    let ids: Vec<Uuid> = (0..5).map(|_| Uuid::new_v4()).collect();
    let serialized = serialize_memory_id_list(&ids);
    let truncated = &serialized[..serialized.len() - 10]; // Remove 10 bytes
    let _ = deserialize_memory_id_list(truncated);
}

#[test]
#[should_panic(expected = "STORAGE ERROR")]
fn test_panic_on_invalid_fingerprint_key() {
    let _ = parse_fingerprint_key(&[0u8; 15]); // Should be 16
}

#[test]
#[should_panic(expected = "STORAGE ERROR")]
fn test_panic_on_invalid_splade_key() {
    let _ = parse_e13_splade_key(&[0u8; 3]); // Should be 2
}

// =========================================================================
// Content Key Panic Tests
// =========================================================================

#[test]
#[should_panic(expected = "STORAGE ERROR")]
fn test_content_key_parse_invalid_length_panics() {
    let invalid_key = [0u8; 8]; // Only 8 bytes, should be 16
    schema::parse_content_key(&invalid_key);
}

