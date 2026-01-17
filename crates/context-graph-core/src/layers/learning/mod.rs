//! L4 Learning Layer - UTL-driven weight optimization.
//!
//! The Learning layer implements the Unified Theory of Learning (UTL) weight
//! optimization formula: W' = W + η*(S⊗C_w)
//!
//! # Constitution Compliance
//!
//! - Latency budget: <10ms
//! - Frequency: 100Hz
//! - Gradient clipping: 1.0
//! - Components: UTL optimizer, neuromod controller
//! - UTL: L optimization (weight updates based on surprise × coherence)
//!
//! # Critical Rules
//!
//! - NO BACKWARDS COMPATIBILITY: System works or fails fast
//! - NO MOCK DATA: Returns real weight updates or proper errors
//! - NO FALLBACKS: If UTL computation fails, ERROR OUT
//!
//! # UTL Weight Update Formula
//!
//! The canonical weight update: W' = W + η*(S⊗C_w)
//! Where:
//! - W = current weight
//! - η = learning rate (0.0005 from constitution)
//! - S = surprise signal (from L1 delta_s × L3 novelty)
//! - C_w = weighted coherence (from pulse)
//! - ⊗ = element-wise product (Hadamard, scalar for global signal)

mod constants;
mod layer;
mod utl_computer;
mod weight_delta;

#[cfg(test)]
mod tests;

// Re-export all public items for backwards compatibility
#[allow(deprecated)]
pub use constants::DEFAULT_CONSOLIDATION_THRESHOLD;
pub use constants::{DEFAULT_LEARNING_RATE, GRADIENT_CLIP, TARGET_FREQUENCY_HZ};
pub use layer::LearningLayer;
pub use utl_computer::UtlWeightComputer;
pub use weight_delta::WeightDelta;
