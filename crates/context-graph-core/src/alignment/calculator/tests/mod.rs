//! Tests for the alignment calculator module.

mod alignment_tests;
mod multi_space_tests;
mod pattern_tests;
mod similarity_tests;

use crate::alignment::calculator::*;
use crate::alignment::config::AlignmentConfig;
use crate::alignment::misalignment::MisalignmentFlags;
use crate::alignment::score::{GoalAlignmentScore, GoalScore, LevelWeights};
use crate::purpose::{GoalDiscoveryMetadata, GoalHierarchy, GoalLevel, GoalNode};
use crate::types::fingerprint::{
    PurposeVector, SemanticFingerprint, TeleologicalFingerprint, NUM_EMBEDDERS,
};

pub(crate) fn test_discovery() -> GoalDiscoveryMetadata {
    GoalDiscoveryMetadata::bootstrap()
}

/// Create a SemanticFingerprint with deterministic values based on a seed.
pub(crate) fn create_test_semantic_fingerprint(seed: f32) -> SemanticFingerprint {
    let mut fp = SemanticFingerprint::zeroed();

    // E1: Semantic (1024D)
    for i in 0..fp.e1_semantic.len() {
        fp.e1_semantic[i] = ((i as f32 / 128.0 + seed).sin()).clamp(-1.0, 1.0);
    }

    // E2: Temporal Recent (512D)
    for i in 0..fp.e2_temporal_recent.len() {
        fp.e2_temporal_recent[i] = ((i as f32 / 64.0 + seed * 1.1).sin()).clamp(-1.0, 1.0);
    }

    // E3: Temporal Periodic (512D)
    for i in 0..fp.e3_temporal_periodic.len() {
        fp.e3_temporal_periodic[i] = ((i as f32 / 64.0 + seed * 1.2).sin()).clamp(-1.0, 1.0);
    }

    // E4: Temporal Positional (512D)
    for i in 0..fp.e4_temporal_positional.len() {
        fp.e4_temporal_positional[i] = ((i as f32 / 64.0 + seed * 1.3).sin()).clamp(-1.0, 1.0);
    }

    // E5: Causal (768D)
    for i in 0..fp.e5_causal.len() {
        fp.e5_causal[i] = ((i as f32 / 96.0 + seed * 1.4).sin()).clamp(-1.0, 1.0);
    }

    // E6: Sparse - add some values
    fp.e6_sparse.indices = vec![10, 50, 100, 200];
    fp.e6_sparse.values = vec![seed.abs() * 0.5 + 0.1; 4];

    // E7: Code (1536D)
    for i in 0..fp.e7_code.len() {
        fp.e7_code[i] = ((i as f32 / 192.0 + seed * 1.5).sin()).clamp(-1.0, 1.0);
    }

    // E8: Graph (384D)
    for i in 0..fp.e8_graph.len() {
        fp.e8_graph[i] = ((i as f32 / 48.0 + seed * 1.6).sin()).clamp(-1.0, 1.0);
    }

    // E9: HDC (1024D)
    for i in 0..fp.e9_hdc.len() {
        fp.e9_hdc[i] = ((i as f32 / 128.0 + seed * 1.7).sin()).clamp(-1.0, 1.0);
    }

    // E10: Multimodal (768D)
    for i in 0..fp.e10_multimodal.len() {
        fp.e10_multimodal[i] = ((i as f32 / 96.0 + seed * 1.8).sin()).clamp(-1.0, 1.0);
    }

    // E11: Entity (384D)
    for i in 0..fp.e11_entity.len() {
        fp.e11_entity[i] = ((i as f32 / 48.0 + seed * 1.9).sin()).clamp(-1.0, 1.0);
    }

    // E12: Late Interaction - add a few token vectors
    fp.e12_late_interaction = vec![
        (0..128)
            .map(|i| ((i as f32 / 16.0 + seed * 2.0).sin()).clamp(-1.0, 1.0))
            .collect(),
        (0..128)
            .map(|i| ((i as f32 / 16.0 + seed * 2.1).sin()).clamp(-1.0, 1.0))
            .collect(),
    ];

    // E13: SPLADE sparse
    fp.e13_splade.indices = vec![5, 25, 75, 150];
    fp.e13_splade.values = vec![seed.abs() * 0.4 + 0.2; 4];

    fp
}

pub(crate) fn create_test_fingerprint(alignment: f32) -> TeleologicalFingerprint {
    let semantic = create_test_semantic_fingerprint(alignment);
    let purpose_vector = PurposeVector::new([alignment; NUM_EMBEDDERS]);

    TeleologicalFingerprint {
        id: uuid::Uuid::new_v4(),
        semantic,
        purpose_vector,
        purpose_evolution: Vec::new(),
        alignment_score: alignment,
        content_hash: [0u8; 32],
        created_at: chrono::Utc::now(),
        last_updated: chrono::Utc::now(),
        access_count: 0,
    }
}

/// TASK-P0-001: Updated for 3-level hierarchy (Strategic → Tactical → Immediate)
pub(crate) fn create_test_hierarchy() -> GoalHierarchy {
    let mut hierarchy = GoalHierarchy::new();

    // Strategic goal (top-level, no parent)
    let s1 = GoalNode::autonomous_goal(
        "Build the best product".into(),
        GoalLevel::Strategic,
        create_test_semantic_fingerprint(0.8),
        test_discovery(),
    )
    .expect("FAIL: Could not create Strategic goal");
    let s1_id = s1.id;
    hierarchy
        .add_goal(s1)
        .expect("FAIL: Could not add Strategic goal to hierarchy");

    // Tactical goal (child of Strategic)
    let t1 = GoalNode::child_goal(
        "Improve user experience".into(),
        GoalLevel::Tactical,
        s1_id,
        create_test_semantic_fingerprint(0.75),
        test_discovery(),
    )
    .expect("FAIL: Could not create Tactical goal");
    let t1_id = t1.id;
    hierarchy
        .add_goal(t1)
        .expect("FAIL: Could not add Tactical goal to hierarchy");

    // Tactical goal 2 (another child of Strategic)
    let t2 = GoalNode::child_goal(
        "Reduce page load time".into(),
        GoalLevel::Tactical,
        s1_id,
        create_test_semantic_fingerprint(0.7),
        test_discovery(),
    )
    .expect("FAIL: Could not create Tactical goal 2");
    hierarchy
        .add_goal(t2)
        .expect("FAIL: Could not add Tactical goal 2 to hierarchy");

    // Immediate goal (child of Tactical)
    let i1 = GoalNode::child_goal(
        "Optimize image loading".into(),
        GoalLevel::Immediate,
        t1_id,
        create_test_semantic_fingerprint(0.65),
        test_discovery(),
    )
    .expect("FAIL: Could not create Immediate goal");
    hierarchy
        .add_goal(i1)
        .expect("FAIL: Could not add Immediate goal to hierarchy");

    hierarchy
}
