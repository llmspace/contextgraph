//! Consciousness State Machine
//!
//! Implements state transitions for consciousness levels as specified in
//! Constitution v4.0.0 Section gwt.state_machine (lines 394-408).
//!
//! ## States
//!
//! - **DORMANT**: r < 0.3, no active workspace
//! - **FRAGMENTED**: 0.3 ≤ r < 0.5, partial synchronization
//! - **EMERGING**: 0.5 ≤ r < 0.8, approaching consciousness
//! - **CONSCIOUS**: r ≥ 0.8, unified perception
//! - **HYPERSYNC**: r > 0.95, pathological overdrive (warning state)

mod manager;
mod types;

#[cfg(test)]
mod tests;

// Re-export all public types for backwards compatibility
pub use manager::StateMachineManager;
pub use types::{ConsciousnessState, StateTransition, TransitionAnalysis};
