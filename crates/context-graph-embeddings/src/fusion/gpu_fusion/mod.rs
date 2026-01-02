//! GPU-accelerated fusion layer components for RTX 5090.
//!
//! This module provides GPU implementations of the FuseMoE fusion layer
//! using Candle for CUDA acceleration.
//!
//! # Architecture
//!
//! ```text
//! Input: [batch_size, 8320] (concatenated embeddings)
//!        |
//!        v
//!   [GpuLayerNorm(8320)] -----> Normalize to mean=0, var=1
//!        |
//!        v
//!   [GpuLinear(8320 -> 8)] ----> Expert routing logits
//!        |
//!        v
//!   [Temperature Softmax] -----> Expert probabilities
//!        |
//!        v
//!   [Top-K Selection] ---------> (indices, weights)
//!        |
//!        v
//!   [GpuExpertPool] -----------> Weighted expert outputs
//!        |
//!        v
//!   Output: [batch_size, 1536] (fused embedding)
//! ```
//!
//! # Hardware Target
//!
//! - NVIDIA RTX 5090 32GB (Blackwell GB202)
//! - CUDA 13.1 with Compute Capability 12.0
//! - Expected speedup: 60-100x vs CPU
//!
//! # No Fallbacks Policy
//!
//! All GPU operations fail fast with descriptive errors.
//! No CPU fallbacks are implemented - system must work or fail for debugging.

mod activation;
mod expert;
mod expert_pool;
mod fusemoe;
mod gating;
mod layer_norm;
mod linear;

#[cfg(test)]
mod tests;
#[cfg(test)]
mod tests_expert;
#[cfg(test)]
mod tests_pool;

// Re-export all public types for backwards compatibility
#[cfg(feature = "candle")]
pub use activation::GpuActivation;
#[cfg(feature = "candle")]
pub use expert::GpuExpert;
#[cfg(feature = "candle")]
pub use expert_pool::GpuExpertPool;
#[cfg(feature = "candle")]
pub use fusemoe::GpuFuseMoE;
#[cfg(feature = "candle")]
pub use gating::GpuGatingNetwork;
#[cfg(feature = "candle")]
pub use layer_norm::GpuLayerNorm;
#[cfg(feature = "candle")]
pub use linear::GpuLinear;
