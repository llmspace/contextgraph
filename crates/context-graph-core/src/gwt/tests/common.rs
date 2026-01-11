//! Common test utilities and helper functions for GWT tests

use crate::types::fingerprint::{
    JohariFingerprint, PurposeVector, SemanticFingerprint, TeleologicalFingerprint,
};

/// Helper to create a test TeleologicalFingerprint with known alignments
pub fn create_test_fingerprint(alignments: [f32; 13]) -> TeleologicalFingerprint {
    let purpose_vector = PurposeVector::new(alignments);
    let semantic = SemanticFingerprint::zeroed();
    let johari = JohariFingerprint::zeroed();

    TeleologicalFingerprint {
        id: uuid::Uuid::new_v4(),
        semantic,
        purpose_vector,
        johari,
        purpose_evolution: Vec::new(),
        theta_to_north_star: alignments.iter().sum::<f32>() / 13.0,
        content_hash: [0u8; 32],
        created_at: chrono::Utc::now(),
        last_updated: chrono::Utc::now(),
        access_count: 0,
    }
}
