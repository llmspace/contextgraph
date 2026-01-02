//! Binary format constants for FusedEmbedding serialization.

use crate::types::dimensions::{FUSED_OUTPUT, NUM_EXPERTS, TOP_K_EXPERTS};

/// Binary format constants.
/// Core embedding size: 1536*4 + 8*4 + 4 + 8 + 8 + 4 = 6200 bytes.
pub const VECTOR_BYTES: usize = FUSED_OUTPUT * 4;
pub const WEIGHTS_BYTES: usize = NUM_EXPERTS * 4;
pub const SELECTED_BYTES: usize = TOP_K_EXPERTS;
pub const LATENCY_BYTES: usize = 8;
pub const HASH_BYTES: usize = 8;
pub const AUX_LEN_BYTES: usize = 4;
pub const CORE_BINARY_SIZE: usize =
    VECTOR_BYTES + WEIGHTS_BYTES + SELECTED_BYTES + LATENCY_BYTES + HASH_BYTES + AUX_LEN_BYTES;

/// Tolerance for expert weight sum validation.
pub const WEIGHT_SUM_TOLERANCE: f32 = 0.01;
