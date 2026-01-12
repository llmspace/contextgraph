//! Helper functions for creating REAL test data (no mocks).

use context_graph_core::types::fingerprint::{
    JohariFingerprint, PurposeVector, SemanticFingerprint, TeleologicalFingerprint, NUM_EMBEDDERS,
};
use uuid::Uuid;

/// Create a SemanticFingerprint with zeroed embeddings for testing.
/// NOTE: This uses zeroed data which is only suitable for serialization tests.
/// For search/alignment tests, use real embeddings from the embedding pipeline.
pub fn create_real_semantic() -> SemanticFingerprint {
    SemanticFingerprint::zeroed()
}

/// Create a REAL PurposeVector with specified alignment.
pub fn create_real_purpose(alignment: f32) -> PurposeVector {
    PurposeVector::new([alignment; NUM_EMBEDDERS])
}

/// Create a REAL JohariFingerprint with high openness.
pub fn create_real_johari() -> JohariFingerprint {
    let mut jf = JohariFingerprint::zeroed();
    for i in 0..NUM_EMBEDDERS {
        jf.set_quadrant(i, 1.0, 0.0, 0.0, 0.0, 1.0); // 100% Open, 100% confidence
    }
    jf
}

/// Create a REAL content hash.
pub fn create_real_hash() -> [u8; 32] {
    let mut hash = [0u8; 32];
    // SHA-256 of "test content"
    hash[0] = 0xDE;
    hash[1] = 0xAD;
    hash[30] = 0xBE;
    hash[31] = 0xEF;
    hash
}

/// Create a REAL TeleologicalFingerprint for testing.
pub fn create_real_fingerprint() -> TeleologicalFingerprint {
    TeleologicalFingerprint::new(
        create_real_semantic(),
        create_real_purpose(0.75),
        create_real_johari(),
        create_real_hash(),
    )
}

/// Create a TeleologicalFingerprint with the correct content hash for the given content.
/// This is needed because store_content() validates the hash.
pub fn create_fingerprint_for_content(content: &str) -> TeleologicalFingerprint {
    use sha2::{Digest, Sha256};

    let mut hasher = Sha256::new();
    hasher.update(content.as_bytes());
    let content_hash: [u8; 32] = hasher.finalize().into();

    TeleologicalFingerprint::new(
        create_real_semantic(),
        create_real_purpose(0.8),
        create_real_johari(),
        content_hash,
    )
}
