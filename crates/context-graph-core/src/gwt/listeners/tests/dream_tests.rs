//! Tests for DreamEventListener
//!
//! # Constitution Compliance
//!
//! Per AP-26: These tests use `new_for_testing()` which provides fail-fast
//! behavior. Tests that don't explicitly test IC crisis handling will panic
//! if an IC event triggers a dream (AP-26 enforcement).
//!
//! For IC crisis tests, see the tests in `dream.rs` which use the full
//! `new()` constructor with proper TriggerManager and callback.

use std::sync::Arc;
use tokio::sync::RwLock;
use uuid::Uuid;

use crate::gwt::listeners::DreamEventListener;
use crate::gwt::workspace::{WorkspaceEvent, WorkspaceEventListener};
use chrono::Utc;

// ============================================================
// FSV Tests for DreamEventListener
// ============================================================

#[tokio::test]
async fn test_fsv_dream_listener_memory_exits() {
    println!("=== FSV: DreamEventListener - MemoryExits ===");

    // SETUP - use new_for_testing since this test doesn't test IC handling
    let dream_queue = Arc::new(RwLock::new(Vec::new()));
    let listener = DreamEventListener::new_for_testing(dream_queue.clone());
    let memory_id = Uuid::new_v4();

    // BEFORE
    let before_len = {
        let queue = dream_queue.read().await;
        queue.len()
    };
    println!("BEFORE: queue.len() = {}", before_len);
    assert_eq!(before_len, 0, "Queue must start empty");

    // EXECUTE
    let event = WorkspaceEvent::MemoryExits {
        id: memory_id,
        order_parameter: 0.65,
        timestamp: Utc::now(),
    };
    listener.on_event(&event);

    // AFTER - SEPARATE READ
    let after_len = {
        let queue = dream_queue.read().await;
        queue.len()
    };
    let queued_id = {
        let queue = dream_queue.read().await;
        queue.first().cloned()
    };
    println!("AFTER: queue.len() = {}", after_len);

    // VERIFY
    assert_eq!(after_len, 1, "Queue must have exactly 1 item");
    assert_eq!(queued_id, Some(memory_id), "Queued ID must match");

    // EVIDENCE
    println!(
        "EVIDENCE: Memory {:?} correctly queued for dream replay",
        memory_id
    );
}

#[tokio::test]
async fn test_dream_listener_ignores_other_events() {
    println!("=== TEST: DreamEventListener ignores non-MemoryExits ===");

    // Use new_for_testing since this test doesn't test IC handling
    let dream_queue = Arc::new(RwLock::new(Vec::new()));
    let listener = DreamEventListener::new_for_testing(dream_queue.clone());

    // Send MemoryEnters - should be ignored
    let event = WorkspaceEvent::MemoryEnters {
        id: Uuid::new_v4(),
        order_parameter: 0.85,
        timestamp: Utc::now(),
        fingerprint: None, // TASK-IDENTITY-P0-006
    };
    listener.on_event(&event);

    // Send WorkspaceEmpty - should be ignored
    let event = WorkspaceEvent::WorkspaceEmpty {
        duration_ms: 1000,
        timestamp: Utc::now(),
    };
    listener.on_event(&event);

    let queue_len = {
        let queue = dream_queue.read().await;
        queue.len()
    };

    assert_eq!(
        queue_len, 0,
        "Queue should remain empty for non-MemoryExits events"
    );
    println!("EVIDENCE: DreamEventListener correctly ignores non-MemoryExits events");
}

#[tokio::test]
async fn test_dream_listener_identity_critical_above_threshold() {
    println!("=== TEST: DreamEventListener handles IdentityCritical (above threshold) ===");

    // Use new_for_testing - this is safe because IC=0.7 is above threshold (0.5)
    // and won't trigger the callback (which would panic)
    let dream_queue = Arc::new(RwLock::new(Vec::new()));
    let listener = DreamEventListener::new_for_testing(dream_queue.clone());

    // Send IdentityCritical with IC above threshold - should NOT trigger dream
    // AP-26: IC=0.7 > 0.5 threshold, so no callback invocation
    let event = WorkspaceEvent::IdentityCritical {
        identity_coherence: 0.7, // Above threshold 0.5
        previous_status: "Stable".to_string(),
        current_status: "Warning".to_string(),
        reason: "Test warning (above threshold)".to_string(),
        timestamp: Utc::now(),
    };
    listener.on_event(&event);

    let queue_len = {
        let queue = dream_queue.read().await;
        queue.len()
    };

    assert_eq!(
        queue_len, 0,
        "Queue should remain empty for IdentityCritical"
    );
    println!("EVIDENCE: IdentityCritical event handled without queuing (IC above threshold)");
}
