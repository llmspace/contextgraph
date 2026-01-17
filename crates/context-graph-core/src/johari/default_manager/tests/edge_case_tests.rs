//! Edge case tests for DefaultJohariManager (Required by TASK-L004).

use crate::johari::default_manager::DefaultJohariManager;
use crate::johari::error::JohariError;
use crate::johari::manager::{ClassificationContext, JohariTransitionManager};
use crate::stubs::InMemoryTeleologicalStore;
use crate::traits::TeleologicalMemoryStore;
use crate::types::fingerprint::{
    JohariFingerprint, PurposeVector, SemanticFingerprint, TeleologicalFingerprint, NUM_EMBEDDERS,
};
use crate::types::{JohariQuadrant, TransitionTrigger};
use std::sync::Arc;

fn create_test_store() -> Arc<InMemoryTeleologicalStore> {
    Arc::new(InMemoryTeleologicalStore::new())
}

fn create_test_fingerprint() -> TeleologicalFingerprint {
    TeleologicalFingerprint::new(
        SemanticFingerprint::zeroed(),
        PurposeVector::default(),
        JohariFingerprint::zeroed(),
        [0u8; 32],
    )
}

#[tokio::test]
async fn edge_case_1_empty_signals() {
    let store = create_test_store();
    let manager = DefaultJohariManager::new(store.clone());

    let fp = create_test_fingerprint();
    let id = store.store(fp).await.unwrap();

    let before = store.retrieve(id).await.unwrap().unwrap();
    let result = manager.discover_blind_spots(id, &[]).await.unwrap();

    assert!(result.is_empty());

    let after = store.retrieve(id).await.unwrap().unwrap();
    assert_eq!(before.johari.quadrants, after.johari.quadrants);
}

#[tokio::test]
async fn edge_case_2_boundary_classification() {
    let store = create_test_store();
    let manager = DefaultJohariManager::new(store);

    let semantic = SemanticFingerprint::zeroed();
    let context = ClassificationContext {
        delta_s: [0.5; NUM_EMBEDDERS],
        delta_c: [0.5; NUM_EMBEDDERS],
        disclosure_intent: [true; NUM_EMBEDDERS],
        access_counts: [0; NUM_EMBEDDERS],
    };

    let result = manager.classify(&semantic, &context).await.unwrap();

    // High S + Low C = Blind
    for i in 0..NUM_EMBEDDERS {
        assert_eq!(result.dominant_quadrant(i), JohariQuadrant::Blind);
    }
}

#[tokio::test]
async fn edge_case_3_max_embedder_index() {
    let store = create_test_store();
    let manager = DefaultJohariManager::new(store.clone());

    let mut fp = create_test_fingerprint();
    for i in 0..NUM_EMBEDDERS {
        fp.johari.set_quadrant(i, 0.0, 0.0, 0.0, 1.0, 1.0);
    }
    let id = store.store(fp).await.unwrap();

    // Index 12 should work (valid max)
    let valid_result = manager
        .transition(
            id,
            12,
            JohariQuadrant::Open,
            TransitionTrigger::DreamConsolidation,
        )
        .await;

    assert!(valid_result.is_ok());

    // Index 13 should fail
    let invalid_result = manager
        .transition(
            id,
            13,
            JohariQuadrant::Open,
            TransitionTrigger::DreamConsolidation,
        )
        .await;

    assert!(invalid_result.is_err());
    match invalid_result {
        Err(JohariError::InvalidEmbedderIndex(idx)) => {
            assert_eq!(idx, 13);
        }
        other => panic!("Expected InvalidEmbedderIndex, got {:?}", other),
    }
}
