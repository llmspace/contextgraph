//! L5 Coherence Layer - Per-space clustering coordination and Global Workspace broadcast.
//!
//! The Coherence layer implements Global Workspace Theory (GWT) with per-space
//! clustering coordination for coherent memory integration.
//!
//! # Constitution Compliance (v6.0.0)
//!
//! - Latency budget: <10ms
//! - Throughput: 100/s
//! - Components: Per-space clustering, GW broadcast, workspace update
//! - UTL: R(t) measurement (coherence)
//!
//! # Critical Rules
//!
//! - NO BACKWARDS COMPATIBILITY: System works or fails fast
//! - NO MOCK DATA: Returns real coherence or proper errors
//! - NO FALLBACKS: If coherence computation fails, ERROR OUT
//!
//! # Topic-Based Coherence Scoring
//!
//! Per Constitution v6.0.0 Section 14, coherence scoring replaces GWT consciousness:
//!
//! coherence_score = I(t) × R(t) × D(t)
//!
//! Where:
//! - I(t) = Integration (information available for global broadcast)
//! - R(t) = Resonance (coherence from per-space clustering)
//! - D(t) = Differentiation (normalized Shannon entropy of purpose vector)

mod constants;
mod layer;
mod thresholds;
mod workspace;

#[cfg(test)]
mod tests;

// Re-export new thresholds module
pub use thresholds::GwtThresholds;

// Re-export constants (deprecated re-exports with warnings)
#[allow(deprecated)]
pub use constants::{
    FRAGMENTATION_THRESHOLD, GW_THRESHOLD, HYPERSYNC_THRESHOLD, INTEGRATION_STEPS,
};

// Re-export layer components
pub use layer::CoherenceLayer;
pub use workspace::{CoherenceState, GlobalWorkspace};
