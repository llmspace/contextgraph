//! Transition tests for DefaultJohariManager.

use crate::johari::default_manager::DefaultJohariManager;
use crate::johari::error::JohariError;
use crate::johari::external_signal::ExternalSignal;
use crate::johari::manager::JohariTransitionManager;
use crate::stubs::InMemoryTeleologicalStore;
use crate::traits::TeleologicalMemoryStore;
use crate::types::fingerprint::{
    JohariFingerprint, PurposeVector, SemanticFingerprint, TeleologicalFingerprint,
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
async fn test_transition_valid() {
    let store = create_test_store();
    let manager = DefaultJohariManager::new(store.clone());

    let mut fp = create_test_fingerprint();
    fp.johari.set_quadrant(0, 0.0, 1.0, 0.0, 0.0, 1.0);
    let id = store.store(fp).await.unwrap();

    let result = manager
        .transition(id, 0, JohariQuadrant::Open, TransitionTrigger::ExplicitShare)
        .await
        .unwrap();

    assert_eq!(result.dominant_quadrant(0), JohariQuadrant::Open);

    let stored = store.retrieve(id).await.unwrap().unwrap();
    assert_eq!(stored.johari.dominant_quadrant(0), JohariQuadrant::Open);
}

#[tokio::test]
async fn test_transition_invalid_returns_error() {
    let store = create_test_store();
    let manager = DefaultJohariManager::new(store.clone());

    let mut fp = create_test_fingerprint();
    fp.johari.set_quadrant(0, 1.0, 0.0, 0.0, 0.0, 1.0);
    let id = store.store(fp).await.unwrap();

    let result = manager
        .transition(id, 0, JohariQuadrant::Blind, TransitionTrigger::ExternalObservation)
        .await;

    assert!(result.is_err());
    match result.unwrap_err() {
        JohariError::InvalidTransition { from, to, embedder_idx } => {
            assert_eq!(from, JohariQuadrant::Open);
            assert_eq!(to, JohariQuadrant::Blind);
            assert_eq!(embedder_idx, 0);
        }
        e => panic!("Expected InvalidTransition, got {:?}", e),
    }
}

#[tokio::test]
async fn test_discover_blind_spots() {
    let store = create_test_store();
    let manager = DefaultJohariManager::new(store.clone()).with_blind_spot_threshold(0.5);

    let mut fp = create_test_fingerprint();
    fp.johari.set_quadrant(5, 0.0, 0.0, 0.0, 1.0, 1.0);
    let id = store.store(fp).await.unwrap();

    let signals = vec![
        ExternalSignal::new("user_feedback", 5, 0.4),
        ExternalSignal::new("dream_layer", 5, 0.3),
    ];

    let candidates = manager.discover_blind_spots(id, &signals).await.unwrap();

    assert_eq!(candidates.len(), 1);
    assert_eq!(candidates[0].embedder_idx, 5);
    assert_eq!(candidates[0].current_quadrant, JohariQuadrant::Unknown);
    assert!((candidates[0].signal_strength - 0.7).abs() < 0.01);
}

#[tokio::test]
async fn test_batch_transition_all_or_nothing() {
    let store = create_test_store();
    let manager = DefaultJohariManager::new(store.clone());

    let mut fp = create_test_fingerprint();
    for i in 0..5 {
        fp.johari.set_quadrant(i, 0.0, 0.0, 0.0, 1.0, 1.0);
    }
    let id = store.store(fp).await.unwrap();

    let transitions = vec![
        (0, JohariQuadrant::Open, TransitionTrigger::DreamConsolidation),
        (99, JohariQuadrant::Open, TransitionTrigger::DreamConsolidation),
    ];

    let result = manager.transition_batch(id, transitions).await;
    assert!(result.is_err());

    let stored = store.retrieve(id).await.unwrap().unwrap();
    assert_eq!(stored.johari.dominant_quadrant(0), JohariQuadrant::Unknown);
}

#[tokio::test]
async fn test_batch_transition_success() {
    let store = create_test_store();
    let manager = DefaultJohariManager::new(store.clone());

    let mut fp = create_test_fingerprint();
    for i in 0..5 {
        fp.johari.set_quadrant(i, 0.0, 0.0, 0.0, 1.0, 1.0);
    }
    let id = store.store(fp).await.unwrap();

    let transitions = vec![
        (0, JohariQuadrant::Open, TransitionTrigger::DreamConsolidation),
        (1, JohariQuadrant::Hidden, TransitionTrigger::DreamConsolidation),
        (2, JohariQuadrant::Blind, TransitionTrigger::ExternalObservation),
    ];

    let result = manager.transition_batch(id, transitions).await.unwrap();

    assert_eq!(result.dominant_quadrant(0), JohariQuadrant::Open);
    assert_eq!(result.dominant_quadrant(1), JohariQuadrant::Hidden);
    assert_eq!(result.dominant_quadrant(2), JohariQuadrant::Blind);
}
