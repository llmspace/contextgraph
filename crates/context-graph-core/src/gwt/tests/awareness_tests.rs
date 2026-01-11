//! Self-awareness loop activation tests for GwtSystem
//!
//! TASK-GWT-P0-003: Tests for:
//! - Self-awareness loop field existence
//! - process_action_awareness updates
//! - Full state verification

use super::common::create_test_fingerprint;
use crate::gwt::{GwtSystem, IdentityStatus};
use std::time::Duration;

// ============================================================
// Test: GwtSystem has self_awareness_loop field
// ============================================================
#[tokio::test]
async fn test_gwt_system_has_self_awareness_loop() {
    println!("=== TEST: GwtSystem has self_awareness_loop field ===");

    let gwt = GwtSystem::new().await.expect("GwtSystem must create");

    // Verify field exists and is accessible
    let loop_mgr = gwt.self_awareness_loop.read().await;
    let ic = loop_mgr.identity_coherence();
    let status = loop_mgr.identity_status();

    println!("EVIDENCE: self_awareness_loop accessible");
    println!("  - identity_coherence: {:.4}", ic);
    println!("  - identity_status: {:?}", status);

    // Initial state should be Critical (IC = 0.0 < 0.5)
    assert_eq!(
        status,
        IdentityStatus::Critical,
        "Initial identity status must be Critical per constitution.yaml"
    );
    assert_eq!(ic, 0.0, "Initial IC must be 0.0");
}

// ============================================================
// Test: process_action_awareness updates purpose_vector
// ============================================================
#[tokio::test]
async fn test_process_action_awareness_updates_purpose_vector() {
    println!("=== TEST: process_action_awareness updates purpose_vector ===");

    let gwt = GwtSystem::new().await.unwrap();

    // BEFORE: Check initial purpose_vector
    let initial_pv = {
        let ego = gwt.self_ego_node.read().await;
        ego.purpose_vector
    };
    println!("BEFORE: purpose_vector = {:?}", initial_pv);
    assert_eq!(initial_pv, [0.0; 13], "Initial pv must be zeros");

    // Step Kuramoto to get some sync
    for _ in 0..20 {
        gwt.step_kuramoto(Duration::from_millis(10)).await;
    }

    // Create fingerprint with known alignments
    let alignments = [
        0.8, 0.75, 0.9, 0.6, 0.7, 0.65, 0.85, 0.72, 0.78, 0.68, 0.82, 0.71, 0.76,
    ];
    let fingerprint = create_test_fingerprint(alignments);

    // EXECUTE
    let result = gwt.process_action_awareness(&fingerprint).await;
    assert!(result.is_ok(), "process_action_awareness must succeed");
    let reflection_result = result.unwrap();

    // AFTER: Verify purpose_vector was updated
    let final_pv = {
        let ego = gwt.self_ego_node.read().await;
        ego.purpose_vector
    };
    println!("AFTER: purpose_vector = {:?}", final_pv);
    assert_eq!(
        final_pv, alignments,
        "purpose_vector must match fingerprint alignments"
    );

    println!("EVIDENCE: purpose_vector correctly updated via process_action_awareness");
    println!("  - alignment: {:.4}", reflection_result.alignment);
    println!("  - identity_status: {:?}", reflection_result.identity_status);
}

// ============================================================
// Full State Verification: process_action_awareness integration
// ============================================================
#[tokio::test]
async fn test_fsv_process_action_awareness() {
    println!("=== FULL STATE VERIFICATION: process_action_awareness ===");

    // SOURCE OF TRUTH: GwtSystem fields
    let gwt = GwtSystem::new().await.unwrap();

    // Step Kuramoto to establish sync
    for _ in 0..50 {
        gwt.step_kuramoto(Duration::from_millis(10)).await;
    }
    let kuramoto_r = gwt.get_kuramoto_r().await;
    println!("SETUP: kuramoto_r = {:.4}", kuramoto_r);

    // BEFORE state
    println!("\nSTATE BEFORE:");
    {
        let ego = gwt.self_ego_node.read().await;
        println!("  - purpose_vector[0]: {:.4}", ego.purpose_vector[0]);
        println!("  - coherence_with_actions: {:.4}", ego.coherence_with_actions);
        println!(
            "  - identity_trajectory.len: {}",
            ego.identity_trajectory.len()
        );
    }
    {
        let loop_mgr = gwt.self_awareness_loop.read().await;
        println!("  - identity_coherence: {:.4}", loop_mgr.identity_coherence());
        println!("  - identity_status: {:?}", loop_mgr.identity_status());
    }

    // Create high-alignment fingerprint
    let alignments = [0.85; 13];
    let fingerprint = create_test_fingerprint(alignments);

    // EXECUTE
    let result = gwt
        .process_action_awareness(&fingerprint)
        .await
        .expect("process_action_awareness must succeed");

    // AFTER state - VERIFY VIA SEPARATE READS
    println!("\nSTATE AFTER:");
    let final_pv;
    let final_coherence;
    let trajectory_len;
    {
        let ego = gwt.self_ego_node.read().await;
        final_pv = ego.purpose_vector;
        final_coherence = ego.coherence_with_actions;
        trajectory_len = ego.identity_trajectory.len();
        println!("  - purpose_vector[0]: {:.4}", final_pv[0]);
        println!("  - coherence_with_actions: {:.4}", final_coherence);
        println!("  - identity_trajectory.len: {}", trajectory_len);
    }
    {
        let loop_mgr = gwt.self_awareness_loop.read().await;
        println!("  - identity_coherence: {:.4}", loop_mgr.identity_coherence());
        println!("  - identity_status: {:?}", loop_mgr.identity_status());
    }

    // ASSERTIONS
    assert_eq!(final_pv, alignments, "purpose_vector must match input");
    assert!(final_coherence > 0.0, "coherence must be updated");
    assert!(trajectory_len > 0, "identity_trajectory must have snapshot");

    println!("\nRESULT:");
    println!("  - alignment: {:.4}", result.alignment);
    println!("  - needs_reflection: {}", result.needs_reflection);
    println!("  - identity_status: {:?}", result.identity_status);
    println!("  - identity_coherence: {:.4}", result.identity_coherence);

    println!("\nEVIDENCE OF SUCCESS: All state fields correctly updated");
}
