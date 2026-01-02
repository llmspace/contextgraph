//! FuseMoE configuration constants.
//!
//! These constants define the expert network configuration for the fusion layer.

/// Number of expert networks in FuseMoE.
/// Constitution.yaml specifies 8 experts for the fusion layer.
pub const NUM_EXPERTS: usize = 8;

/// Top-K experts selected for each input (routing).
/// Constitution.yaml specifies top_k=4 for sparse expert activation.
pub const TOP_K_EXPERTS: usize = 4;

/// ColBERT v3 per-token embedding dimension.
/// Used for AuxiliaryEmbeddingData in FusedEmbedding.
pub const COLBERT_V3_DIM: usize = 128;

// =============================================================================
// COMPILE-TIME VALIDATION
// =============================================================================

/// Compile-time assertion that NUM_EXPERTS equals 8.
const _NUM_EXPERTS_CHECK: () = assert!(NUM_EXPERTS == 8, "NUM_EXPERTS must equal 8");

/// Compile-time assertion that TOP_K_EXPERTS equals 4.
const _TOP_K_EXPERTS_CHECK: () = assert!(TOP_K_EXPERTS == 4, "TOP_K_EXPERTS must equal 4");

/// Compile-time assertion that COLBERT_V3_DIM equals 128.
const _COLBERT_V3_DIM_CHECK: () = assert!(COLBERT_V3_DIM == 128, "COLBERT_V3_DIM must equal 128");

/// Compile-time assertion that TOP_K_EXPERTS < NUM_EXPERTS.
const _TOP_K_LESS_THAN_NUM_CHECK: () = assert!(
    TOP_K_EXPERTS < NUM_EXPERTS,
    "TOP_K_EXPERTS must be less than NUM_EXPERTS"
);
