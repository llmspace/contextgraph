//! Gating network for FuseMoE routing.
//!
//! This module implements the gating network that routes the 8320D concatenated
//! embeddings to 8 experts based on temperature-scaled softmax probabilities.
//!
//! # Architecture
//!
//! ```text
//! Input: [batch_size, 8320]
//!        |
//!        v
//!   [LayerNorm(8320)]
//!        |
//!        v
//!   [Linear(8320 -> 8)]
//!        |
//!        v
//!   [Temperature-scaled Softmax] --> Expert weights [batch_size, 8]
//!        |
//!        v
//!   [Laplace Smoothing (optional)]
//!        |
//!        v
//!   [Top-K Selection] --> Selected experts and weights
//! ```
//!
//! # Constitution Compliance
//!
//! - `num_experts = 8` (constitution.yaml: fuse_moe.num_experts)
//! - `top_k = 4` (constitution.yaml: fuse_moe.top_k)
//! - `temperature = 1.0` (default, neuromodulation range [0.5, 2.0])
//! - `laplace_alpha = 0.01` (constitution.yaml: fuse_moe.laplace_alpha)
//!
//! # No Fallbacks Policy
//!
//! - Invalid dimensions -> `EmbeddingError::InvalidDimension`
//! - NaN values -> `EmbeddingError::InvalidValue`
//! - Empty input -> `EmbeddingError::EmptyInput`

mod layer_norm;
mod linear;
mod network;
pub mod routing;
pub mod softmax;

#[cfg(test)]
mod tests;

// Re-export all public types for backwards compatibility
pub use layer_norm::LayerNorm;
pub use linear::Linear;
pub use network::GatingNetwork;
