//! Expert Networks for FuseMoE routing.
//!
//! This module implements 8 expert networks that transform the concatenated
//! 8320D embedding into 1536D outputs. The gating network selects top-k experts
//! and provides routing weights for weighted combination.
//!
//! # Architecture
//!
//! ```text
//! Expert FFN: input(8320) -> hidden(4096) -> GELU -> output(1536)
//!
//! ExpertPool Flow:
//! 1. Receive (indices, weights) from GatingNetwork.select_top_k()
//! 2. Forward input through selected experts
//! 3. Compute weighted combination of outputs
//! 4. Return fused 1536D embedding
//! ```
//!
//! # Constitution Compliance
//!
//! - `num_experts = 8` (constitution.yaml: fuse_moe.num_experts)
//! - `expert_hidden_dim = 4096` (FusionConfig)
//! - `output_dim = 1536` (FUSED_OUTPUT constant)
//!
//! # No Fallbacks Policy
//!
//! - Invalid expert index -> `EmbeddingError::InvalidExpertIndex`
//! - Dimension mismatch -> `EmbeddingError::DimensionMismatch`
//! - Empty input -> `EmbeddingError::EmptyInput`
//!
//! # Module Organization
//!
//! - [`activation`]: Activation functions (GELU, ReLU, SiLU)
//! - [`expert`]: Single expert network implementation
//! - [`pool`]: Expert pool with top-k routing

mod activation;
mod expert;
mod pool;

// Re-export all public types for backwards compatibility
pub use activation::Activation;
pub use expert::Expert;
pub use pool::ExpertPool;

#[cfg(test)]
mod tests;
