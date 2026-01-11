//! Meta-Cognitive Feedback Loop
//!
//! Implements self-correction and adaptive learning through meta-cognitive monitoring
//! as specified in Constitution v4.0.0 Section gwt.meta_cognitive (lines 410-417).
//!
//! ## Module Structure
//!
//! - `types`: Type definitions and constants
//! - `core`: Core implementation logic
//!
//! ## Formula
//!
//! MetaScore = σ(2 × (L_predicted - L_actual))
//!
//! Where:
//! - L_predicted: Meta-UTL predicted learning score
//! - L_actual: Actual observed learning score
//! - σ: Sigmoid function
//!
//! ## Self-Correction Protocol
//!
//! - Low MetaScore (<0.5) for 5+ consecutive operations → increase Acetylcholine, trigger dream
//! - High MetaScore (>0.9) → reduce meta-monitoring frequency

mod core;
mod types;

#[cfg(test)]
mod tests;

// Re-export all public types for backwards compatibility
pub use types::{
    FrequencyAdjustment, MetaCognitiveLoop, MetaCognitiveState, NeuromodulationEffect, ScoreTrend,
    ACH_BASELINE, ACH_DECAY_RATE, ACH_MAX,
};
