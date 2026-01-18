//! Helper functions for UTL tests

use context_graph_core::types::fingerprint::{
    PurposeVector, SemanticFingerprint, TeleologicalFingerprint,
};

/// Create a test TeleologicalFingerprint with specified semantic values.
///
/// Uses zeroed base with modified e1_semantic for testing ΔS computation.
pub fn create_test_fingerprint_with_semantic(semantic_values: Vec<f32>) -> TeleologicalFingerprint {
    let mut semantic = SemanticFingerprint::zeroed();
    semantic.e1_semantic = semantic_values;
    TeleologicalFingerprint::new(semantic, PurposeVector::default(), [0u8; 32])
}
