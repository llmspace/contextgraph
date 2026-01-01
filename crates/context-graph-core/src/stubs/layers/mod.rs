//! Stub implementations of NervousLayer for all 5 bio-nervous system layers.
//!
//! These implementations provide deterministic, instant responses for the
//! Ghost System phase (Phase 0). Production implementations will replace
//! these with real processing logic.
//!
//! Each stub:
//! - Returns immediately (no sleep)
//! - Reports duration well within latency budget
//! - Returns deterministic output based on input
//! - Always passes health_check()
//!
//! # Module Structure
//! - `helpers` - Common types and functions
//! - `sensing` - L1 Sensing layer (5ms budget)
//! - `reflex` - L2 Reflex layer (100us budget)
//! - `memory` - L3 Memory layer (1ms budget)
//! - `learning` - L4 Learning layer (10ms budget)
//! - `coherence` - L5 Coherence layer (10ms budget)

mod coherence;
pub mod helpers;
mod learning;
mod memory;
mod reflex;
mod sensing;

#[cfg(test)]
mod tests_edge_cases;
#[cfg(test)]
mod tests_integration;
#[cfg(test)]
mod tests_latency;

// Re-export all stub layers for backwards compatibility
pub use coherence::StubCoherenceLayer;
pub use learning::StubLearningLayer;
pub use memory::StubMemoryLayer;
pub use reflex::StubReflexLayer;
pub use sensing::StubSensingLayer;
