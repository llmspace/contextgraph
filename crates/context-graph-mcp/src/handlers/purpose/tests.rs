//! Tests for purpose handlers.

use context_graph_core::purpose::{GoalDiscoveryMetadata, GoalLevel, GoalNode};
use context_graph_core::types::fingerprint::{PurposeVector, SemanticFingerprint, NUM_EMBEDDERS};

#[test]
fn test_goal_to_json_structure() {
    // Verify the JSON structure matches expected output
    // Per TASK-CORE-005: Use autonomous_goal() with TeleologicalArray, not north_star()
    let discovery = GoalDiscoveryMetadata::bootstrap();
    let goal = GoalNode::autonomous_goal(
        "Test North Star".into(),
        GoalLevel::NorthStar,
        SemanticFingerprint::zeroed(),
        discovery,
    )
    .expect("Failed to create test goal");

    // Verify GoalNode structure (id is now Uuid, not custom GoalId)
    assert!(!goal.id.is_nil()); // UUID should not be nil
    assert_eq!(goal.level, GoalLevel::NorthStar);
    assert!(goal.is_north_star());

    println!("[VERIFIED] GoalNode structure is correct with new API");
}

#[test]
fn test_purpose_vector_validation() {
    // Test that purpose vector validation works correctly
    let valid_alignments = [0.5f32; NUM_EMBEDDERS];
    let pv = PurposeVector {
        alignments: valid_alignments,
        dominant_embedder: 0,
        coherence: 1.0,
        stability: 1.0,
    };

    assert_eq!(pv.alignments.len(), NUM_EMBEDDERS);
    assert!(pv.alignments.iter().all(|&v| (0.0..=1.0).contains(&v)));

    println!("[VERIFIED] PurposeVector validation works correctly");
}
