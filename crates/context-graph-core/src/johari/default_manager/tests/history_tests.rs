//! Transition history tests for DefaultJohariManager.
//!
//! These tests verify the transition history recording functionality,
//! which was a critical bug fix.

use crate::johari::default_manager::DefaultJohariManager;
use crate::johari::manager::{JohariTransitionManager, TimeRange};
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

    // Store a fingerprint with Hidden quadrant
    let mut fp = create_test_fingerprint();
    fp.johari.set_quadrant(0, 0.0, 1.0, 0.0, 0.0, 1.0); // Hidden
    let id = store.store(fp).await.unwrap();

    // Perform a transition
    let _result = manager
        .transition(
            id,
            0,
            JohariQuadrant::Open,
            TransitionTrigger::ExplicitShare,
        )
        .await
        .unwrap();

    // Get transition history
    let history = manager.get_transition_history(id, 10).await.unwrap();

    // [VERIFY] Transition was recorded
    assert_eq!(history.len(), 1, "Expected 1 transition in history");
    assert_eq!(history[0].memory_id, id);
    assert_eq!(history[0].embedder_idx, 0);
    assert_eq!(history[0].from, JohariQuadrant::Hidden);
    assert_eq!(history[0].to, JohariQuadrant::Open);
    assert_eq!(history[0].trigger, TransitionTrigger::ExplicitShare);

    println!(
        "[VERIFIED] test_transition_history_recorded: Transition correctly recorded in history"
    );
}

#[tokio::test]
async fn test_transition_stats_computed() {
    let store = create_test_store();
    let manager = DefaultJohariManager::new(store.clone());

    // Store a fingerprint with Hidden quadrant
    let mut fp = create_test_fingerprint();
    fp.johari.set_quadrant(0, 0.0, 1.0, 0.0, 0.0, 1.0); // Hidden
    let id = store.store(fp).await.unwrap();

    // Perform a transition
    let _result = manager
        .transition(
            id,
            0,
            JohariQuadrant::Open,
            TransitionTrigger::ExplicitShare,
        )
        .await
        .unwrap();

    // Get transition stats for last 24 hours
    let time_range = TimeRange::last_hours(24);
    let stats = manager.get_transition_stats(time_range).await.unwrap();

    // [VERIFY] Stats are computed from real transitions
    assert_eq!(stats.total_transitions, 1, "Expected 1 total transition");
    assert_eq!(stats.memories_affected, 1, "Expected 1 memory affected");
    assert_eq!(
        stats.count_for_path(JohariQuadrant::Hidden, JohariQuadrant::Open),
        1,
        "Expected 1 Hidden->Open transition"
    );
    assert_eq!(
        stats.count_for_trigger(TransitionTrigger::ExplicitShare),
        1,
        "Expected 1 ExplicitShare trigger"
    );
    assert_eq!(
        stats.count_for_embedder(0),
        1,
        "Expected 1 transition on E1"
    );

    println!(
        "[VERIFIED] test_transition_stats_computed: Stats correctly computed from real transitions"
    );
}

#[tokio::test]
async fn test_batch_transitions_recorded_in_history() {
    let store = create_test_store();
    let manager = DefaultJohariManager::new(store.clone());

    // Store with multiple Unknown embedders
    let mut fp = create_test_fingerprint();
    for i in 0..5 {
        fp.johari.set_quadrant(i, 0.0, 0.0, 0.0, 1.0, 1.0); // Unknown
    }
    let id = store.store(fp).await.unwrap();

    // Perform batch transitions
    let transitions = vec![
        (
            0,
            JohariQuadrant::Open,
            TransitionTrigger::DreamConsolidation,
        ),
        (
            1,
            JohariQuadrant::Hidden,
            TransitionTrigger::DreamConsolidation,
        ),
        (
            2,
            JohariQuadrant::Blind,
            TransitionTrigger::ExternalObservation,
        ),
    ];
    let _result = manager.transition_batch(id, transitions).await.unwrap();

    // Get transition history
    let history = manager.get_transition_history(id, 10).await.unwrap();

    // [VERIFY] All batch transitions were recorded
    assert_eq!(history.len(), 3, "Expected 3 transitions in history");

    // History is in reverse chronological order, so newest first
    // Transitions are inserted in order, so embedder 2 is newest
    let t0 = history
        .iter()
        .find(|t| t.embedder_idx == 0)
        .expect("Missing E1 transition");
    let t1 = history
        .iter()
        .find(|t| t.embedder_idx == 1)
        .expect("Missing E2 transition");
    let t2 = history
        .iter()
        .find(|t| t.embedder_idx == 2)
        .expect("Missing E3 transition");

    assert_eq!(t0.from, JohariQuadrant::Unknown);
    assert_eq!(t0.to, JohariQuadrant::Open);
    assert_eq!(t1.from, JohariQuadrant::Unknown);
    assert_eq!(t1.to, JohariQuadrant::Hidden);
    assert_eq!(t2.from, JohariQuadrant::Unknown);
    assert_eq!(t2.to, JohariQuadrant::Blind);

    println!(
        "[VERIFIED] test_batch_transitions_recorded_in_history: All batch transitions recorded"
    );
}

#[tokio::test]
async fn test_history_filtered_by_memory_id() {
    let store = create_test_store();
    let manager = DefaultJohariManager::new(store.clone());

    // Store two fingerprints
    let mut fp1 = create_test_fingerprint();
    fp1.johari.set_quadrant(0, 0.0, 1.0, 0.0, 0.0, 1.0); // Hidden
    let id1 = store.store(fp1).await.unwrap();

    let mut fp2 = create_test_fingerprint();
    fp2.johari.set_quadrant(0, 0.0, 1.0, 0.0, 0.0, 1.0); // Hidden
    let id2 = store.store(fp2).await.unwrap();

    // Perform transitions on both
    let _result = manager
        .transition(
            id1,
            0,
            JohariQuadrant::Open,
            TransitionTrigger::ExplicitShare,
        )
        .await
        .unwrap();
    let _result = manager
        .transition(
            id2,
            0,
            JohariQuadrant::Open,
            TransitionTrigger::ExplicitShare,
        )
        .await
        .unwrap();

    // Get history for just id1
    let history1 = manager.get_transition_history(id1, 10).await.unwrap();
    let history2 = manager.get_transition_history(id2, 10).await.unwrap();

    // [VERIFY] Each memory has only its own transitions
    assert_eq!(history1.len(), 1);
    assert_eq!(history1[0].memory_id, id1);
    assert_eq!(history2.len(), 1);
    assert_eq!(history2[0].memory_id, id2);

    println!(
        "[VERIFIED] test_history_filtered_by_memory_id: History correctly filtered per memory"
    );
}

#[tokio::test]
async fn test_multiple_transitions_same_memory() {
    let store = create_test_store();
    let manager = DefaultJohariManager::new(store.clone());

    // Store a fingerprint with Unknown quadrants
    let mut fp = create_test_fingerprint();
    for i in 0..NUM_EMBEDDERS {
        fp.johari.set_quadrant(i, 0.0, 0.0, 0.0, 1.0, 1.0); // Unknown
    }
    let id = store.store(fp).await.unwrap();

    // Perform multiple transitions on different embedders
    for i in 0..3 {
        let _result = manager
            .transition(
                id,
                i,
                JohariQuadrant::Open,
                TransitionTrigger::DreamConsolidation,
            )
            .await
            .unwrap();
    }

    // Get transition history
    let history = manager.get_transition_history(id, 10).await.unwrap();

    // [VERIFY] All transitions were recorded
    assert_eq!(history.len(), 3, "Expected 3 transitions in history");

    // Verify they are in reverse chronological order (newest first)
    // Since we inserted 0, 1, 2 in order, they should be 2, 1, 0 when retrieved
    assert_eq!(history[0].embedder_idx, 2);
    assert_eq!(history[1].embedder_idx, 1);
    assert_eq!(history[2].embedder_idx, 0);

    println!(
        "[VERIFIED] test_multiple_transitions_same_memory: Multiple transitions correctly recorded and ordered"
    );
}

#[tokio::test]
async fn test_history_limit_respected() {
    let store = create_test_store();
    let manager = DefaultJohariManager::new(store.clone());

    // Store a fingerprint with Unknown quadrants
    let mut fp = create_test_fingerprint();
    for i in 0..NUM_EMBEDDERS {
        fp.johari.set_quadrant(i, 0.0, 0.0, 0.0, 1.0, 1.0); // Unknown
    }
    let id = store.store(fp).await.unwrap();

    // Perform 5 transitions
    for i in 0..5 {
        let _result = manager
            .transition(
                id,
                i,
                JohariQuadrant::Open,
                TransitionTrigger::DreamConsolidation,
            )
            .await
            .unwrap();
    }

    // Get transition history with limit of 2
    let history = manager.get_transition_history(id, 2).await.unwrap();

    // [VERIFY] Only 2 transitions returned
    assert_eq!(history.len(), 2, "Expected only 2 transitions due to limit");

    // Should be the 2 most recent (newest first)
    assert_eq!(history[0].embedder_idx, 4);
    assert_eq!(history[1].embedder_idx, 3);

    println!("[VERIFIED] test_history_limit_respected: Limit correctly applied to history query");
}
