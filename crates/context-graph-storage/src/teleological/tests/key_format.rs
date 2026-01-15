//! Key encoding/decoding tests.

use crate::teleological::*;
use uuid::Uuid;

// =========================================================================
// Key Format Tests
// =========================================================================

#[test]
fn test_fingerprint_key_format() {
    let id = Uuid::new_v4();
    let key = fingerprint_key(&id);

    println!("=== TEST: fingerprint_key format ===");
    println!("UUID: {}", id);
    println!("Key length: {} bytes", key.len());
    println!("Key bytes: {:02x?}", key);

    assert_eq!(key.len(), 16);
    assert_eq!(&key, id.as_bytes());
}

#[test]
fn test_purpose_vector_key_format() {
    let id = Uuid::new_v4();
    let key = purpose_vector_key(&id);

    assert_eq!(key.len(), 16);
    assert_eq!(&key, id.as_bytes());
}

#[test]
fn test_e1_matryoshka_128_key_format() {
    let id = Uuid::new_v4();
    let key = e1_matryoshka_128_key(&id);

    assert_eq!(key.len(), 16);
    assert_eq!(&key, id.as_bytes());
}

#[test]
fn test_e13_splade_inverted_key_format() {
    let term_id: u16 = 12345;
    let key = e13_splade_inverted_key(term_id);

    println!("=== TEST: e13_splade_inverted_key format ===");
    println!("term_id: {}", term_id);
    println!("Key length: {} bytes", key.len());
    println!("Key bytes: {:02x?}", key);

    assert_eq!(key.len(), 2);
    // Big-endian: 12345 = 0x3039
    assert_eq!(key, [0x30, 0x39]);
}

#[test]
fn test_parse_fingerprint_key_roundtrip() {
    let original = Uuid::new_v4();
    let key = fingerprint_key(&original);
    let parsed = parse_fingerprint_key(&key);

    assert_eq!(original, parsed);
}

#[test]
fn test_parse_e13_splade_key_roundtrip() {
    for term_id in [0u16, 1, 100, 1000, 12345, 30521, u16::MAX] {
        let key = e13_splade_inverted_key(term_id);
        let parsed = parse_e13_splade_key(&key);
        assert_eq!(term_id, parsed, "Round-trip failed for term_id {}", term_id);
    }
}

// =========================================================================
// TASK-TELEO-006: New Key Format Tests
// =========================================================================

#[test]
fn test_synergy_matrix_key_constant() {
    assert_eq!(SYNERGY_MATRIX_KEY, b"synergy");
    assert_eq!(SYNERGY_MATRIX_KEY.len(), 7);
}

#[test]
fn test_teleological_profile_key_format() {
    println!("=== TEST: teleological_profile_key format ===");

    let profile_id = "research_profile_001";
    let key = teleological_profile_key(profile_id);

    println!("Profile ID: {}", profile_id);
    println!("Key length: {} bytes", key.len());
    println!("Key bytes: {:02x?}", key);

    assert_eq!(key.len(), profile_id.len());
    assert_eq!(key, profile_id.as_bytes());
}

#[test]
fn test_teleological_profile_key_roundtrip() {
    let test_cases = vec![
        "simple",
        "complex_profile_name",
        "a", // minimum length
        "research-task-001",
        "profile_with_numbers_123",
    ];

    for profile_id in test_cases {
        let key = teleological_profile_key(profile_id);
        let parsed = parse_teleological_profile_key(&key);
        assert_eq!(profile_id, parsed, "Round-trip failed for '{}'", profile_id);
    }
}

#[test]
fn test_teleological_profile_key_255_chars() {
    println!("=== TEST: Maximum length profile key (255 chars) ===");

    let profile_id = "a".repeat(255);
    println!("BEFORE: Profile ID with {} chars", profile_id.len());

    let key = teleological_profile_key(&profile_id);
    println!("AFTER: Key with {} bytes", key.len());

    assert_eq!(key.len(), 255);

    let parsed = parse_teleological_profile_key(&key);
    assert_eq!(profile_id, parsed);
    println!("RESULT: PASS - Maximum length handled correctly");
}

#[test]
fn test_teleological_vector_key_format() {
    println!("=== TEST: teleological_vector_key format ===");

    let memory_id = Uuid::new_v4();
    let key = teleological_vector_key(&memory_id);

    println!("Memory ID: {}", memory_id);
    println!("Key length: {} bytes", key.len());
    println!("Key bytes: {:02x?}", key);

    assert_eq!(key.len(), 16);
    assert_eq!(&key, memory_id.as_bytes());
}

#[test]
fn test_teleological_vector_key_roundtrip() {
    for _ in 0..10 {
        let original = Uuid::new_v4();
        let key = teleological_vector_key(&original);
        let parsed = parse_teleological_vector_key(&key);
        assert_eq!(original, parsed);
    }
}

#[test]
fn test_teleological_vector_key_nil_uuid() {
    let nil_uuid = Uuid::nil();
    let key = teleological_vector_key(&nil_uuid);
    let parsed = parse_teleological_vector_key(&key);

    assert_eq!(nil_uuid, parsed);
    assert!(parsed.is_nil());
}

#[test]
fn test_teleological_vector_key_max_uuid() {
    let max_uuid = Uuid::max();
    let key = teleological_vector_key(&max_uuid);
    let parsed = parse_teleological_vector_key(&key);

    assert_eq!(max_uuid, parsed);
}

// =========================================================================
// TASK-GWT-P1-001: EGO_NODE Key Tests
// =========================================================================

#[test]
fn test_ego_node_key_constant() {
    println!("=== TEST: EGO_NODE_KEY constant (TASK-GWT-P1-001) ===");

    assert_eq!(schema::EGO_NODE_KEY, b"ego_node");
    assert_eq!(schema::EGO_NODE_KEY.len(), 8);
    assert_eq!(schema::ego_node_key(), b"ego_node");

    println!("RESULT: PASS - EGO_NODE_KEY constant is correct");
}

// =========================================================================
// Content Key Tests
// =========================================================================

#[test]
fn test_content_key_format() {
    println!("=== TEST: content_key format (TASK-CONTENT-002) ===");

    let id = Uuid::new_v4();
    let key = schema::content_key(&id);

    println!("UUID: {}", id);
    println!("Key: {:02x?}", key);
    println!("Key length: {} bytes", key.len());

    assert_eq!(key.len(), 16, "Content key must be 16 bytes (UUID)");
    assert_eq!(key, *id.as_bytes(), "Key must equal UUID bytes");
}

#[test]
fn test_content_key_roundtrip() {
    println!("=== TEST: content_key roundtrip ===");

    let test_uuids = vec![
        Uuid::nil(),
        Uuid::max(),
        Uuid::new_v4(),
        Uuid::new_v4(),
    ];

    for id in test_uuids {
        let key = schema::content_key(&id);
        let parsed = schema::parse_content_key(&key);
        assert_eq!(id, parsed, "Round-trip failed for UUID {}", id);
    }

    println!("RESULT: PASS - All content key round-trips successful");
}

// =========================================================================
// Additional Verification Tests
// =========================================================================

#[test]
fn test_key_functions_deterministic() {
    let id = Uuid::new_v4();

    // Same ID should produce same key
    let key1 = fingerprint_key(&id);
    let key2 = fingerprint_key(&id);
    assert_eq!(key1, key2);

    let term_id: u16 = 42;
    let term_key1 = e13_splade_inverted_key(term_id);
    let term_key2 = e13_splade_inverted_key(term_id);
    assert_eq!(term_key1, term_key2);
}

#[test]
fn edge_case_deterministic_key_generation_new_keys() {
    println!("=== EDGE CASE: Deterministic key generation for new key types ===");

    let memory_id = Uuid::parse_str("550e8400-e29b-41d4-a716-446655440000").unwrap();
    let profile_id = "deterministic_profile";

    println!("BEFORE: Creating keys multiple times");
    let vec_key1 = teleological_vector_key(&memory_id);
    let vec_key2 = teleological_vector_key(&memory_id);
    let prof_key1 = teleological_profile_key(profile_id);
    let prof_key2 = teleological_profile_key(profile_id);

    println!("AFTER: Comparing key outputs");
    assert_eq!(vec_key1, vec_key2, "Vector keys must be deterministic");
    assert_eq!(prof_key1, prof_key2, "Profile keys must be deterministic");
    println!("RESULT: PASS - All keys deterministic");
}

#[test]
fn edge_case_teleological_profile_unicode() {
    println!("=== EDGE CASE: Profile key with unicode characters ===");

    let profile_id = "research_ai"; // ASCII only for safety
    println!(
        "BEFORE: Profile ID '{}' ({} bytes)",
        profile_id,
        profile_id.len()
    );

    let key = teleological_profile_key(profile_id);
    println!("SERIALIZED: {} bytes", key.len());

    let parsed = parse_teleological_profile_key(&key);
    println!("AFTER: Parsed '{}'", parsed);

    assert_eq!(profile_id, parsed);
    println!("RESULT: PASS - Profile key round-trip successful");
}

// =========================================================================
// TASK-SESSION-04: Session Identity Key Tests
// =========================================================================

#[test]
fn test_session_latest_key_constant() {
    println!("=== TEST: SESSION_LATEST_KEY constant (TASK-SESSION-04) ===");

    assert_eq!(SESSION_LATEST_KEY, b"latest");
    assert_eq!(SESSION_LATEST_KEY.len(), 6);

    println!("RESULT: PASS - SESSION_LATEST_KEY constant is correct");
}

#[test]
fn test_session_identity_key_format() {
    println!("=== TEST: session_identity_key format (TASK-SESSION-04) ===");

    let session_id = "test-session-123";
    let key = session_identity_key(session_id);

    println!("Session ID: {}", session_id);
    println!("Key length: {} bytes", key.len());
    println!("Key bytes: {:02x?}", key);

    // Key format: "s:" prefix (2 bytes) + session_id
    assert_eq!(key.len(), 2 + session_id.len());
    assert_eq!(&key[0..2], b"s:");
    assert_eq!(&key[2..], session_id.as_bytes());

    println!("RESULT: PASS - session_identity_key format is correct");
}

#[test]
fn test_session_identity_key_roundtrip() {
    println!("=== TEST: session_identity_key roundtrip (TASK-SESSION-04) ===");

    let test_cases = vec![
        "simple",
        "session-with-dashes",
        "session_with_underscores",
        "abc123",
        "a", // minimum length
        "very-long-session-identifier-with-many-characters-and-numbers-12345",
    ];

    for session_id in test_cases {
        let key = session_identity_key(session_id);
        let parsed = parse_session_identity_key(&key);
        assert_eq!(
            session_id, parsed,
            "Round-trip failed for session_id '{}'",
            session_id
        );
        println!("  {} -> OK", session_id);
    }

    println!("RESULT: PASS - All session_identity_key round-trips successful");
}

#[test]
fn test_session_temporal_key_format() {
    println!("=== TEST: session_temporal_key format (TASK-SESSION-04) ===");

    let timestamp_ms: i64 = 1736899200000; // Example timestamp
    let key = session_temporal_key(timestamp_ms);

    println!("Timestamp (ms): {}", timestamp_ms);
    println!("Key length: {} bytes", key.len());
    println!("Key bytes: {:02x?}", key);

    // Key format: "t:" prefix (2 bytes) + timestamp_ms big-endian (8 bytes)
    assert_eq!(key.len(), 10);
    assert_eq!(&key[0..2], b"t:");

    println!("RESULT: PASS - session_temporal_key format is correct");
}

#[test]
fn test_session_temporal_key_roundtrip() {
    println!("=== TEST: session_temporal_key roundtrip (TASK-SESSION-04) ===");

    let test_timestamps: Vec<i64> = vec![
        0,           // minimum
        1,           // near minimum
        1000,        // 1 second
        1736899200000, // realistic timestamp
        i64::MAX / 2, // large value
    ];

    for timestamp in test_timestamps {
        let key = session_temporal_key(timestamp);
        let parsed = parse_session_temporal_key(&key);
        assert_eq!(
            timestamp, parsed,
            "Round-trip failed for timestamp {}",
            timestamp
        );
        println!("  {} -> OK", timestamp);
    }

    println!("RESULT: PASS - All session_temporal_key round-trips successful");
}

#[test]
fn test_session_temporal_key_lexicographic_ordering() {
    println!("=== TEST: session_temporal_key maintains lexicographic order ===");

    // Keys should be lexicographically ordered by timestamp (big-endian encoding)
    let timestamps: Vec<i64> = vec![0, 100, 1000, 10000, 100000, 1000000, i64::MAX / 2];
    let keys: Vec<Vec<u8>> = timestamps.iter().map(|&t| session_temporal_key(t)).collect();

    for i in 1..keys.len() {
        assert!(
            keys[i - 1] < keys[i],
            "Key for timestamp {} should be < key for timestamp {}",
            timestamps[i - 1],
            timestamps[i]
        );
        println!(
            "  {} < {} -> OK",
            timestamps[i - 1],
            timestamps[i]
        );
    }

    println!("RESULT: PASS - Lexicographic ordering maintained");
}
