//! Panic behavior tests (should_panic).

use crate::teleological::*;
use super::helpers::create_real_fingerprint;
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
fn test_panic_on_wrong_purpose_vector_size() {
    let _ = deserialize_purpose_vector(&[0u8; 51]); // Should be 52
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
// TASK-TELEO-006: Panic Tests for New Key Types
// =========================================================================

#[test]
#[should_panic(expected = "STORAGE ERROR: teleological_profile_key cannot be empty")]
fn test_teleological_profile_key_empty_panics() {
    let _ = teleological_profile_key("");
}

#[test]
#[should_panic(expected = "STORAGE ERROR: teleological_profile_key too long")]
fn test_teleological_profile_key_too_long_panics() {
    let long_id = "x".repeat(256);
    let _ = teleological_profile_key(&long_id);
}

#[test]
#[should_panic(expected = "STORAGE ERROR: teleological_profile key cannot be empty")]
fn test_parse_teleological_profile_key_empty_panics() {
    let _ = parse_teleological_profile_key(&[]);
}

#[test]
#[should_panic(expected = "STORAGE ERROR: teleological_profile key too long")]
fn test_parse_teleological_profile_key_too_long_panics() {
    let long_key = vec![0x61u8; 256]; // 'a' * 256
    let _ = parse_teleological_profile_key(&long_key);
}

#[test]
#[should_panic(expected = "STORAGE ERROR: teleological_vector key must be 16 bytes")]
fn test_parse_teleological_vector_key_too_short_panics() {
    let _ = parse_teleological_vector_key(&[0u8; 15]);
}

#[test]
#[should_panic(expected = "STORAGE ERROR: teleological_vector key must be 16 bytes")]
fn test_parse_teleological_vector_key_too_long_panics() {
    let _ = parse_teleological_vector_key(&[0u8; 17]);
}

#[test]
#[should_panic(expected = "STORAGE ERROR: teleological_vector key must be 16 bytes")]
fn test_parse_teleological_vector_key_empty_panics() {
    let _ = parse_teleological_vector_key(&[]);
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

// =========================================================================
// TASK-GWT-P1-001: EGO_NODE Panic Tests
// =========================================================================

#[test]
#[should_panic(expected = "DESERIALIZATION ERROR")]
fn test_ego_node_deserialize_empty_panics() {
    let _ = serialization::deserialize_ego_node(&[]);
}

#[test]
#[should_panic(expected = "DESERIALIZATION ERROR")]
fn test_ego_node_deserialize_wrong_version_panics() {
    let mut data = vec![255u8]; // Wrong version
    data.extend(vec![0u8; 100]); // Garbage payload
    let _ = serialization::deserialize_ego_node(&data);
}
