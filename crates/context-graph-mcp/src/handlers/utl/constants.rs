//! Constants for UTL computation handlers.
//!
//! TASK-S005: Embedder names and constitution.yaml targets.

use context_graph_core::johari::NUM_EMBEDDERS;

/// Embedder names for trajectory reporting.
/// 13 embedders: E1-E13.
pub(super) const EMBEDDER_NAMES: [&str; NUM_EMBEDDERS] = [
    "semantic",       // E1
    "episodic",       // E2
    "procedural",     // E3
    "emotional",      // E4
    "temporal",       // E5
    "causal",         // E6
    "analogical",     // E7
    "contextual",     // E8
    "hierarchical",   // E9
    "associative",    // E10
    "metacognitive",  // E11
    "intentional",    // E12
    "sparse_lexical", // E13 (SPLADE)
];

/// Constitution.yaml targets (hardcoded per TASK-S005 spec).
pub(super) const LEARNING_SCORE_TARGET: f32 = 0.6;
pub(super) const COHERENCE_RECOVERY_TARGET_MS: u64 = 10000;
pub(super) const ATTACK_DETECTION_TARGET: f32 = 0.95;
pub(super) const FALSE_POSITIVE_TARGET: f32 = 0.02;

/// ΔC formula weights per constitution.yaml line 166.
pub(super) const ALPHA: f32 = 0.4; // Connectivity weight
pub(super) const BETA: f32 = 0.4;  // ClusterFit weight
pub(super) const GAMMA: f32 = 0.2; // Consistency weight
