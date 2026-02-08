//! Key format functions for teleological storage.
//!
//! All keys use fixed-size formats for efficient range scans.
//!
//! # Key Formats (TASK-CONTENT-001)
//!
//! | CF | Key Format | Size |
//! |----|------------|------|
//! | content | fingerprint_id UUID | 16 bytes |
//!
//! # FAIL FAST Policy
//!
//! Key parsing functions panic on invalid input. This ensures:
//! 1. Data corruption is immediately detected
//! 2. No silent degradation of data integrity
//! 3. Clear error messages with full context

use uuid::Uuid;

/// Key for fingerprints CF: UUID as 16 bytes.
///
/// # Arguments
/// * `id` - The fingerprint's UUID
///
/// # Returns
/// Exactly 16 bytes (UUID in big-endian format)
#[inline]
pub fn fingerprint_key(id: &Uuid) -> [u8; 16] {
    *id.as_bytes()
}

/// Key for topic profile storage (CF_TOPIC_PROFILES): UUID as 16 bytes.
///
/// # Arguments
/// * `id` - The fingerprint's UUID
///
/// # Returns
/// Exactly 16 bytes (UUID in big-endian format)
#[inline]
pub fn topic_profile_key(id: &Uuid) -> [u8; 16] {
    *id.as_bytes()
}

/// Key for e13_splade_inverted CF: term_id as 2 bytes (big-endian).
///
/// # Arguments
/// * `term_id` - The SPLADE vocabulary term index (0..30522)
///
/// # Returns
/// Exactly 2 bytes (u16 in big-endian format)
#[inline]
pub fn e13_splade_inverted_key(term_id: u16) -> [u8; 2] {
    term_id.to_be_bytes()
}

/// Key for e6_sparse_inverted CF: term_id as 2 bytes (big-endian).
///
/// Used for E6 (V_selectivity) sparse inverted index to enable:
/// - Stage 1: Dual sparse recall with E13 SPLADE
/// - Stage 3.5: Tie-breaker for close E1 scores
///
/// # Arguments
/// * `term_id` - The BERT vocabulary term index (0..30522, same vocab as E13)
///
/// # Returns
/// Exactly 2 bytes (u16 in big-endian format)
#[inline]
pub fn e6_sparse_inverted_key(term_id: u16) -> [u8; 2] {
    term_id.to_be_bytes()
}

/// Parse E6 sparse inverted key back to term_id.
///
/// # Arguments
/// * `key` - Exactly 2 bytes
///
/// # Returns
/// The parsed term_id (u16)
///
/// # Panics
/// Panics if key is not exactly 2 bytes (FAIL FAST).
#[inline]
pub fn parse_e6_sparse_key(key: &[u8]) -> u16 {
    if key.len() != 2 {
        panic!(
            "STORAGE ERROR: e6_sparse key must be 2 bytes, got {} bytes. \
             Key data: {:02x?}. This indicates corrupted storage or wrong CF access.",
            key.len(),
            key
        );
    }
    u16::from_be_bytes([key[0], key[1]])
}

/// Key for e1_matryoshka_128 CF: UUID as 16 bytes.
///
/// # Arguments
/// * `id` - The fingerprint's UUID
///
/// # Returns
/// Exactly 16 bytes (UUID in big-endian format)
#[inline]
pub fn e1_matryoshka_128_key(id: &Uuid) -> [u8; 16] {
    *id.as_bytes()
}

/// Parse fingerprint key back to UUID.
///
/// # Arguments
/// * `key` - Exactly 16 bytes
///
/// # Returns
/// The parsed UUID
///
/// # Panics
/// Panics if key is not exactly 16 bytes (FAIL FAST).
/// Error message includes:
/// - Actual key length
/// - Key data (for debugging)
/// - Context about what went wrong
#[inline]
pub fn parse_fingerprint_key(key: &[u8]) -> Uuid {
    if key.len() != 16 {
        panic!(
            "STORAGE ERROR: fingerprint key must be 16 bytes, got {} bytes. \
             Key data: {:02x?}. This indicates corrupted storage or wrong CF access.",
            key.len(),
            key
        );
    }
    Uuid::from_slice(key).unwrap_or_else(|e| {
        panic!(
            "STORAGE ERROR: Invalid UUID bytes in fingerprint key. \
             Error: {}. Key data: {:02x?}. This should never happen with valid 16-byte input.",
            e, key
        );
    })
}

/// Parse E13 SPLADE inverted key back to term_id.
///
/// # Arguments
/// * `key` - Exactly 2 bytes
///
/// # Returns
/// The parsed term_id (u16)
///
/// # Panics
/// Panics if key is not exactly 2 bytes (FAIL FAST).
#[inline]
pub fn parse_e13_splade_key(key: &[u8]) -> u16 {
    if key.len() != 2 {
        panic!(
            "STORAGE ERROR: e13_splade key must be 2 bytes, got {} bytes. \
             Key data: {:02x?}. This indicates corrupted storage or wrong CF access.",
            key.len(),
            key
        );
    }
    u16::from_be_bytes([key[0], key[1]])
}

/// Parse topic profile key back to UUID.
///
/// # Arguments
/// * `key` - Exactly 16 bytes
///
/// # Returns
/// The parsed UUID
///
/// # Panics
/// Panics if key is not exactly 16 bytes (FAIL FAST).
#[inline]
pub fn parse_topic_profile_key(key: &[u8]) -> Uuid {
    if key.len() != 16 {
        panic!(
            "STORAGE ERROR: topic_profile key must be 16 bytes, got {} bytes. \
             Key data: {:02x?}. This indicates corrupted storage or wrong CF access.",
            key.len(),
            key
        );
    }
    Uuid::from_slice(key).unwrap_or_else(|e| {
        panic!(
            "STORAGE ERROR: Invalid UUID bytes in topic_profile key. \
             Error: {}. Key data: {:02x?}.",
            e, key
        );
    })
}

/// Parse E1 Matryoshka 128D key back to UUID.
///
/// # Arguments
/// * `key` - Exactly 16 bytes
///
/// # Returns
/// The parsed UUID
///
/// # Panics
/// Panics if key is not exactly 16 bytes (FAIL FAST).
#[inline]
pub fn parse_e1_matryoshka_key(key: &[u8]) -> Uuid {
    if key.len() != 16 {
        panic!(
            "STORAGE ERROR: e1_matryoshka_128 key must be 16 bytes, got {} bytes. \
             Key data: {:02x?}. This indicates corrupted storage or wrong CF access.",
            key.len(),
            key
        );
    }
    Uuid::from_slice(key).unwrap_or_else(|e| {
        panic!(
            "STORAGE ERROR: Invalid UUID bytes in e1_matryoshka_128 key. \
             Error: {}. Key data: {:02x?}.",
            e, key
        );
    })
}

// =============================================================================
// TASK-CONTENT-002: CONTENT KEYS (UUID = 16 bytes)
// =============================================================================

/// Key for content CF: fingerprint_id UUID as 16 bytes.
///
/// Used to store original text content associated with a fingerprint.
/// Same key format as fingerprint_key for consistency.
///
/// # Arguments
/// * `id` - The fingerprint's UUID
///
/// # Returns
/// Exactly 16 bytes (UUID in big-endian format)
#[inline]
pub fn content_key(id: &Uuid) -> [u8; 16] {
    *id.as_bytes()
}

/// Parse content key back to fingerprint UUID.
///
/// # Arguments
/// * `key` - Exactly 16 bytes
///
/// # Returns
/// The parsed UUID
///
/// # Panics
/// Panics if key is not exactly 16 bytes (FAIL FAST).
#[inline]
pub fn parse_content_key(key: &[u8]) -> Uuid {
    if key.len() != 16 {
        panic!(
            "STORAGE ERROR: content key must be 16 bytes, got {} bytes. \
             Key data: {:02x?}. This indicates corrupted storage or wrong CF access.",
            key.len(),
            key
        );
    }
    Uuid::from_slice(key).unwrap_or_else(|e| {
        panic!(
            "STORAGE ERROR: Invalid UUID bytes in content key. \
             Error: {}. Key data: {:02x?}.",
            e, key
        );
    })
}

// =============================================================================
// SOURCE METADATA KEYS (UUID = 16 bytes)
// =============================================================================

/// Key for source_metadata CF: fingerprint_id UUID as 16 bytes.
///
/// Used to store provenance information (source type, file path, etc.)
/// associated with a fingerprint.
///
/// # Arguments
/// * `id` - The fingerprint's UUID
///
/// # Returns
/// Exactly 16 bytes (UUID in big-endian format)
#[inline]
pub fn source_metadata_key(id: &Uuid) -> [u8; 16] {
    *id.as_bytes()
}

/// Parse source_metadata key back to fingerprint UUID.
///
/// # Arguments
/// * `key` - Exactly 16 bytes
///
/// # Returns
/// The parsed UUID
///
/// # Panics
/// Panics if key is not exactly 16 bytes (FAIL FAST).
#[inline]
pub fn parse_source_metadata_key(key: &[u8]) -> Uuid {
    if key.len() != 16 {
        panic!(
            "STORAGE ERROR: source_metadata key must be 16 bytes, got {} bytes. \
             Key data: {:02x?}. This indicates corrupted storage or wrong CF access.",
            key.len(),
            key
        );
    }
    Uuid::from_slice(key).unwrap_or_else(|e| {
        panic!(
            "STORAGE ERROR: Invalid UUID bytes in source_metadata key. \
             Error: {}. Key data: {:02x?}.",
            e, key
        );
    })
}

// =============================================================================
// TASK-STORAGE-P2-001: E12 LATE INTERACTION TOKEN KEYS (UUID = 16 bytes)
// =============================================================================

/// Key for e12_late_interaction CF: memory_id UUID as 16 bytes.
///
/// Used to store ColBERT token embeddings for MaxSim scoring in Stage 5.
///
/// # Arguments
/// * `id` - The memory's UUID
///
/// # Returns
/// Exactly 16 bytes (UUID in big-endian format)
#[inline]
pub fn e12_late_interaction_key(id: &Uuid) -> [u8; 16] {
    *id.as_bytes()
}

/// Parse e12_late_interaction key back to memory UUID.
///
/// # Arguments
/// * `key` - Exactly 16 bytes
///
/// # Returns
/// The parsed UUID
///
/// # Panics
/// Panics if key is not exactly 16 bytes (FAIL FAST).
#[inline]
pub fn parse_e12_late_interaction_key(key: &[u8]) -> Uuid {
    if key.len() != 16 {
        panic!(
            "STORAGE ERROR: e12_late_interaction key must be 16 bytes, got {} bytes. \
             Key data: {:02x?}. This indicates corrupted storage or wrong CF access.",
            key.len(),
            key
        );
    }
    Uuid::from_slice(key).unwrap_or_else(|e| {
        panic!(
            "STORAGE ERROR: Invalid UUID bytes in e12_late_interaction key. \
             Error: {}. Key data: {:02x?}.",
            e, key
        );
    })
}

// =============================================================================
// TASK-SESSION-04: SESSION KEY HELPERS
// =============================================================================

/// Create temporal index key: `t:{timestamp_ms}` (big-endian for lexicographic ordering)
///
/// Big-endian encoding ensures that keys sort chronologically when iterated.
/// This enables efficient temporal range queries.
///
/// # Arguments
/// * `timestamp_ms` - Unix milliseconds timestamp
///
/// # Returns
/// Key bytes: `t:` prefix (2 bytes) + 8-byte big-endian timestamp (10 bytes total)
///
/// # Example
/// ```ignore
/// let k1 = session_temporal_key(1000);
/// let k2 = session_temporal_key(2000);
/// assert!(k1 < k2); // Lexicographic ordering matches numeric
/// ```
#[inline]
pub fn session_temporal_key(timestamp_ms: i64) -> Vec<u8> {
    let mut key = Vec::with_capacity(10);
    key.extend_from_slice(b"t:");
    key.extend_from_slice(&timestamp_ms.to_be_bytes());
    key
}

/// Parse timestamp from session_temporal_key.
///
/// # Arguments
/// * `key` - Exactly 10 bytes: `t:` prefix + 8-byte big-endian timestamp
///
/// # Returns
/// The parsed timestamp in milliseconds
///
/// # Panics
/// Panics if key doesn't start with `t:` or isn't exactly 10 bytes - FAIL FAST policy.
#[inline]
pub fn parse_session_temporal_key(key: &[u8]) -> i64 {
    assert_eq!(
        key.len(),
        10,
        "STORAGE ERROR: session_temporal_key must be exactly 10 bytes, got {} bytes. \
         Key data: {:02x?}. Expected: 2-byte prefix 't:' + 8-byte timestamp.",
        key.len(),
        key
    );
    assert_eq!(
        &key[0..2],
        b"t:",
        "STORAGE ERROR: session_temporal_key must start with 't:'. \
         Got: {:02x?}. This indicates corrupted storage or wrong CF access.",
        &key[0..2]
    );
    i64::from_be_bytes(
        key[2..10]
            .try_into()
            .expect("timestamp bytes slice is exactly 8 bytes"),
    )
}

// =============================================================================
// CAUSAL RELATIONSHIP KEYS (UUID = 16 bytes)
// =============================================================================

/// Key for causal_relationships CF: causal_relationship_id UUID as 16 bytes.
///
/// Used to store LLM-generated causal relationship descriptions with embeddings.
///
/// # Arguments
/// * `id` - The causal relationship's UUID
///
/// # Returns
/// Exactly 16 bytes (UUID in big-endian format)
#[inline]
pub fn causal_relationship_key(id: &Uuid) -> [u8; 16] {
    *id.as_bytes()
}

/// Parse causal_relationship key back to UUID.
///
/// # Arguments
/// * `key` - Exactly 16 bytes
///
/// # Returns
/// The parsed UUID
///
/// # Panics
/// Panics if key is not exactly 16 bytes (FAIL FAST).
#[inline]
pub fn parse_causal_relationship_key(key: &[u8]) -> Uuid {
    if key.len() != 16 {
        panic!(
            "STORAGE ERROR: causal_relationship key must be 16 bytes, got {} bytes. \
             Key data: {:02x?}. This indicates corrupted storage or wrong CF access.",
            key.len(),
            key
        );
    }
    Uuid::from_slice(key).unwrap_or_else(|e| {
        panic!(
            "STORAGE ERROR: Invalid UUID bytes in causal_relationship key. \
             Error: {}. Key data: {:02x?}.",
            e, key
        );
    })
}

/// Key for causal_by_source CF: source_fingerprint_id UUID as 16 bytes.
///
/// Used for secondary index enabling "find all causal relationships from memory X".
///
/// # Arguments
/// * `source_id` - The source fingerprint's UUID
///
/// # Returns
/// Exactly 16 bytes (UUID in big-endian format)
#[inline]
pub fn causal_by_source_key(source_id: &Uuid) -> [u8; 16] {
    *source_id.as_bytes()
}

/// Parse causal_by_source key back to source fingerprint UUID.
///
/// # Arguments
/// * `key` - Exactly 16 bytes
///
/// # Returns
/// The parsed UUID
///
/// # Panics
/// Panics if key is not exactly 16 bytes (FAIL FAST).
#[inline]
pub fn parse_causal_by_source_key(key: &[u8]) -> Uuid {
    if key.len() != 16 {
        panic!(
            "STORAGE ERROR: causal_by_source key must be 16 bytes, got {} bytes. \
             Key data: {:02x?}. This indicates corrupted storage or wrong CF access.",
            key.len(),
            key
        );
    }
    Uuid::from_slice(key).unwrap_or_else(|e| {
        panic!(
            "STORAGE ERROR: Invalid UUID bytes in causal_by_source key. \
             Error: {}. Key data: {:02x?}.",
            e, key
        );
    })
}
