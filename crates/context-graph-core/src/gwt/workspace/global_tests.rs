//! Tests for GlobalWorkspace WTA selection

use super::*;
use crate::neuromod::NeuromodulationManager;

#[tokio::test]
async fn test_workspace_selection_wta() {
    let mut workspace = GlobalWorkspace::new();

    let id1 = Uuid::new_v4();
    let id2 = Uuid::new_v4();
    let id3 = Uuid::new_v4();

    let candidates = vec![
        (id1, 0.85, 0.5, 0.8),  // score ~ 0.34
        (id2, 0.88, 0.9, 0.88), // score ~ 0.7 (winner)
        (id3, 0.92, 0.6, 0.7),  // score ~ 0.387
    ];

    let winner = workspace.select_winning_memory(candidates).await.unwrap();
    assert_eq!(winner, Some(id2));
    assert_eq!(workspace.active_memory, Some(id2));
}

#[tokio::test]
async fn test_workspace_selection_filters_low_coherence() {
    let mut workspace = GlobalWorkspace::new();

    let id1 = Uuid::new_v4();
    let id2 = Uuid::new_v4();

    let candidates = vec![
        (id1, 0.7, 0.9, 0.88), // Below coherence threshold
        (id2, 0.85, 0.8, 0.8), // Above threshold (winner)
    ];

    let winner = workspace.select_winning_memory(candidates).await.unwrap();
    assert_eq!(winner, Some(id2));
}

#[tokio::test]
async fn test_workspace_no_coherent_candidates() {
    let mut workspace = GlobalWorkspace::new();

    let id1 = Uuid::new_v4();
    let id2 = Uuid::new_v4();

    let candidates = vec![(id1, 0.5, 0.9, 0.88), (id2, 0.6, 0.8, 0.8)];

    let winner = workspace.select_winning_memory(candidates).await.unwrap();
    assert_eq!(winner, None);
}

#[test]
fn test_workspace_conflict_detection() {
    let mut workspace = GlobalWorkspace::new();

    let id1 = Uuid::new_v4();
    let id2 = Uuid::new_v4();

    workspace.candidates = vec![
        WorkspaceCandidate::new(id1, 0.85, 0.9, 0.88).unwrap(),
        WorkspaceCandidate::new(id2, 0.82, 0.85, 0.85).unwrap(),
    ];

    assert!(workspace.has_conflict());
    let conflict = workspace.get_conflict_details();
    assert!(conflict.is_some());
    assert_eq!(conflict.unwrap().len(), 2);
}

#[test]
fn test_workspace_broadcast_duration() {
    let workspace = GlobalWorkspace::new();
    assert!(!workspace.is_broadcasting()); // Not yet broadcasting
}

#[test]
fn test_inhibit_losers() {
    println!("=== FSV: GlobalWorkspace::inhibit_losers ===");

    let mut workspace = GlobalWorkspace::new();
    let mut neuromod = NeuromodulationManager::new();

    let winner_id = Uuid::new_v4();
    let loser1_id = Uuid::new_v4();
    let loser2_id = Uuid::new_v4();

    // Set up candidates (winner + 2 losers)
    workspace.candidates = vec![
        WorkspaceCandidate::new(winner_id, 0.90, 0.85, 0.88).unwrap(), // score = 0.6732
        WorkspaceCandidate::new(loser1_id, 0.85, 0.80, 0.82).unwrap(), // score = 0.5576
        WorkspaceCandidate::new(loser2_id, 0.82, 0.75, 0.78).unwrap(), // score = 0.4797
    ];

    // BEFORE
    let before_da = neuromod.get_hopfield_beta();
    println!("BEFORE: dopamine = {:.3}", before_da);

    // EXECUTE
    let inhibited = workspace.inhibit_losers(winner_id, &mut neuromod).unwrap();

    // AFTER
    let after_da = neuromod.get_hopfield_beta();
    println!("AFTER: dopamine = {:.3}", after_da);
    println!("Inhibited count: {}", inhibited);

    // VERIFY
    assert_eq!(inhibited, 2, "Should inhibit exactly 2 losers");
    assert!(
        after_da < before_da,
        "Dopamine should decrease after inhibition"
    );

    // EVIDENCE
    println!(
        "EVIDENCE: {} losers inhibited, dopamine dropped from {:.3} to {:.3}",
        inhibited, before_da, after_da
    );
}

#[test]
fn test_inhibit_losers_single_winner() {
    println!("=== EDGE CASE: inhibit_losers with only winner ===");

    let mut workspace = GlobalWorkspace::new();
    let mut neuromod = NeuromodulationManager::new();

    let winner_id = Uuid::new_v4();
    workspace.candidates = vec![WorkspaceCandidate::new(winner_id, 0.90, 0.85, 0.88).unwrap()];

    let inhibited = workspace.inhibit_losers(winner_id, &mut neuromod).unwrap();

    assert_eq!(inhibited, 0, "Should inhibit 0 with only winner");
    println!("EVIDENCE: No inhibition when only winner present");
}

#[test]
fn test_inhibit_losers_empty_candidates() {
    println!("=== EDGE CASE: inhibit_losers with no candidates ===");

    let workspace = GlobalWorkspace::new();
    let mut neuromod = NeuromodulationManager::new();

    let winner_id = Uuid::new_v4();
    let inhibited = workspace.inhibit_losers(winner_id, &mut neuromod).unwrap();

    assert_eq!(inhibited, 0, "Should inhibit 0 with no candidates");
    println!("EVIDENCE: No inhibition when no candidates");
}

#[test]
fn test_inhibit_losers_magnitude_calculation() {
    println!("=== TEST: inhibit_losers magnitude calculation ===");

    let mut workspace = GlobalWorkspace::new();
    let mut neuromod = NeuromodulationManager::new();

    let winner_id = Uuid::new_v4();
    let loser_id = Uuid::new_v4();

    // Create loser with score = 0.5 exactly
    // score = r * importance * alignment = 0.82 * 0.76 * 0.8 = 0.49856
    workspace.candidates = vec![
        WorkspaceCandidate::new(winner_id, 0.90, 0.85, 0.88).unwrap(),
        WorkspaceCandidate::new(loser_id, 0.82, 0.76, 0.8).unwrap(),
    ];

    let loser_score = workspace.candidates[1].score;
    println!("Loser score: {:.4}", loser_score);

    let before_da = neuromod.get_hopfield_beta();
    workspace.inhibit_losers(winner_id, &mut neuromod).unwrap();
    let after_da = neuromod.get_hopfield_beta();

    // Expected inhibition magnitude = (1.0 - score) * DA_INHIBITION_FACTOR
    // adjust() applies the delta directly to dopamine
    let expected_delta = (1.0 - loser_score) * DA_INHIBITION_FACTOR;
    let actual_delta = before_da - after_da;

    println!(
        "Expected delta: {:.4}, Actual delta: {:.4}",
        expected_delta, actual_delta
    );

    // Allow small floating point tolerance
    assert!(
        (expected_delta - actual_delta).abs() < 0.0001,
        "Inhibition magnitude mismatch: expected {:.4}, got {:.4}",
        expected_delta,
        actual_delta
    );

    println!("EVIDENCE: Inhibition magnitude correctly calculated");
}
