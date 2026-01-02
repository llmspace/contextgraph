//! Gating network for FuseMoE expert routing.
//!
//! Routes the 8320D concatenated embeddings to 8 experts using
//! temperature-scaled softmax with optional Laplace smoothing.
//!
//! # Module Structure
//!
//! - `core`: Core struct definition and constructors
//! - `forward`: Forward pass implementations and top-K selection

mod core;
mod forward;

// Re-export the main type for backwards compatibility
pub use self::core::GatingNetwork;
