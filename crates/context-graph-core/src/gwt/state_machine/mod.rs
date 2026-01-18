//! Coherence State Machine
//!
//! Implements state transitions for coherence levels as specified in
//! Constitution v6.0.0 Section 14 (Topic-based coherence).
//!
//! ## States
//!
//! - **DORMANT**: r < 0.3, no active workspace
//! - **FRAGMENTED**: 0.3 <= r < 0.5, partial synchronization
//! - **EMERGING**: 0.5 <= r < 0.8, approaching coherence
//! - **STABLE**: r >= 0.8, coherent workspace active
//! - **HYPERSYNC**: r > 0.95, pathological overdrive (warning state)

mod manager;
mod types;

#[cfg(test)]
mod tests;

// Re-export all public types
pub use manager::StateMachineManager;
pub use types::{CoherenceState, StateTransition, TransitionAnalysis};
