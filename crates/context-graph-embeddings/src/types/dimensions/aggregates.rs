//! Aggregate dimensions and compile-time validation.
//!
//! This module defines the total concatenated dimension and fused output dimension.

use super::constants::{
    CAUSAL, CODE, ENTITY, GRAPH, HDC, LATE_INTERACTION, MULTIMODAL, SEMANTIC, SPARSE,
    TEMPORAL_PERIODIC, TEMPORAL_POSITIONAL, TEMPORAL_RECENT,
};

// =============================================================================
// AGGREGATE DIMENSIONS
// =============================================================================

/// Total concatenated dimension: sum of all 12 projected dimensions.
/// This is the input size to FuseMoE gating network.
///
/// Calculated as:
/// 1024 + 512 + 512 + 512 + 768 + 1536 + 768 + 384 + 1024 + 768 + 384 + 128 = 8320
pub const TOTAL_CONCATENATED: usize = SEMANTIC
    + TEMPORAL_RECENT
    + TEMPORAL_PERIODIC
    + TEMPORAL_POSITIONAL
    + CAUSAL
    + SPARSE
    + CODE
    + GRAPH
    + HDC
    + MULTIMODAL
    + ENTITY
    + LATE_INTERACTION;

/// FuseMoE output dimension (final unified embedding).
/// Matches OpenAI ada-002 dimension for downstream compatibility.
pub const FUSED_OUTPUT: usize = 1536;

/// Number of models in the ensemble.
pub const MODEL_COUNT: usize = 12;

// =============================================================================
// COMPILE-TIME VALIDATION
// =============================================================================

/// Compile-time assertion that TOTAL_CONCATENATED equals expected value.
/// This will cause a compilation error if dimensions change incorrectly.
const _TOTAL_CONCATENATED_CHECK: () = assert!(
    TOTAL_CONCATENATED == 8320,
    "TOTAL_CONCATENATED must equal 8320"
);

/// Compile-time assertion that FUSED_OUTPUT equals expected value.
const _FUSED_OUTPUT_CHECK: () = assert!(FUSED_OUTPUT == 1536, "FUSED_OUTPUT must equal 1536");

/// Compile-time assertion that MODEL_COUNT equals 12.
const _MODEL_COUNT_CHECK: () = assert!(MODEL_COUNT == 12, "MODEL_COUNT must equal 12");
