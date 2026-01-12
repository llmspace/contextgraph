//! Transition history tests for DefaultJohariManager.

use crate::johari::default_manager::{DefaultJohariManager, DynDefaultJohariManager};
use crate::johari::manager::{ClassificationContext, JohariTransitionManager, TimeRange};
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
async fn test_transition_history_recorded() {
    let store = create_test_store();
    let manager = DefaultJohariManager::new(store.clone());

    let mut fp = create_test_fingerprint();
    fp.johari.set_quadrant(0, 0.0, 1.0, 0.0, 0.0, 1.0);
    let id = store.store(fp).await.unwrap();

    let _ = manager
        .transition(id, 0, JohariQuadrant::Open, TransitionTrigger::ExplicitShare)
        .await
        .unwrap();

    let history = manager.get_transition_history(id, 10).await.unwrap();

    assert_eq!(history.len(), 1);
    assert_eq!(history[0].memory_id, id);
    assert_eq!(history[0].embedder_idx, 0);
    assert_eq!(history[0].from, JohariQuadrant::Hidden);
    assert_eq!(history[0].to, JohariQuadrant::Open);
}

#[tokio::test]
async fn test_transition_stats_computed() {
    let store = create_test_store();
    let manager = DefaultJohariManager::new(store.clone());

    let mut fp = create_test_fingerprint();
    fp.johari.set_quadrant(0, 0.0, 1.0, 0.0, 0.0, 1.0);
    let id = store.store(fp).await.unwrap();

    let _ = manager
        .transition(id, 0, JohariQuadrant::Open, TransitionTrigger::ExplicitShare)
        .await
        .unwrap();

    let time_range = TimeRange::last_hours(24);
    let stats = manager.get_transition_stats(time_range).await.unwrap();

    assert_eq!(stats.total_transitions, 1);
    assert_eq!(stats.memories_affected, 1);
}

#[tokio::test]
async fn test_batch_transitions_recorded_in_history() {
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
    let _ = manager.transition_batch(id, transitions).await.unwrap();

    let history = manager.get_transition_history(id, 10).await.unwrap();

    assert_eq!(history.len(), 3);
}

#[tokio::test]
async fn test_history_filtered_by_memory_id() {
    let store = create_test_store();
    let manager = DefaultJohariManager::new(store.clone());

    let mut fp1 = create_test_fingerprint();
    fp1.johari.set_quadrant(0, 0.0, 1.0, 0.0, 0.0, 1.0);
    let id1 = store.store(fp1).await.unwrap();

    let mut fp2 = create_test_fingerprint();
    fp2.johari.set_quadrant(0, 0.0, 1.0, 0.0, 0.0, 1.0);
    let id2 = store.store(fp2).await.unwrap();

    let _ = manager
        .transition(id1, 0, JohariQuadrant::Open, TransitionTrigger::ExplicitShare)
        .await
        .unwrap();
    let _ = manager
        .transition(id2, 0, JohariQuadrant::Open, TransitionTrigger::ExplicitShare)
        .await
        .unwrap();

    let history1 = manager.get_transition_history(id1, 10).await.unwrap();
    let history2 = manager.get_transition_history(id2, 10).await.unwrap();

    assert_eq!(history1.len(), 1);
    assert_eq!(history1[0].memory_id, id1);
    assert_eq!(history2.len(), 1);
    assert_eq!(history2[0].memory_id, id2);
}

#[tokio::test]
async fn test_dyn_manager_classify() {
    let store: Arc<dyn TeleologicalMemoryStore> = Arc::new(InMemoryTeleologicalStore::new());
    let manager = DynDefaultJohariManager::new(store);
    let semantic = SemanticFingerprint::zeroed();

    let context = ClassificationContext {
        delta_s: [0.3; NUM_EMBEDDERS],
        delta_c: [0.7; NUM_EMBEDDERS],
        disclosure_intent: [true; NUM_EMBEDDERS],
        access_counts: [0; NUM_EMBEDDERS],
    };

    let result = manager.classify(&semantic, &context).await.unwrap();

    for i in 0..NUM_EMBEDDERS {
        assert_eq!(result.dominant_quadrant(i), JohariQuadrant::Open);
    }
}
