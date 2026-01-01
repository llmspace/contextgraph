//! Johari Window quadrant classification for memory categorization.
//!
//! The Johari Window is a psychological model adapted for knowledge graph
//! classification. Each quadrant determines how memories are retrieved and
//! weighted in the UTL (Unified Theory of Learning) system.
//!
//! # Module Structure
//! - `quadrant`: Core JohariQuadrant enum
//! - `transition`: TransitionTrigger and JohariTransition types
//! - `modality`: Content modality classification

mod modality;
mod quadrant;
mod transition;

#[cfg(test)]
mod tests_modality;
#[cfg(test)]
mod tests_quadrant;
#[cfg(test)]
mod tests_transition;

// Re-export all public types for backwards compatibility
pub use modality::Modality;
pub use quadrant::JohariQuadrant;
pub use transition::{JohariTransition, TransitionTrigger};
