//! NORTH-008: BootstrapService Implementation
//!
//! Service for initializing North Star goal from documentation sources.
//! Extracts goal candidates from project documents and selects the best
//! goal to serve as the North Star for the context graph system.
//!
//! # Architecture
//!
//! The bootstrap process follows these steps:
//! 1. Scan document directory for matching files by extension
//! 2. Extract goal candidates from each document using keyword analysis
//! 3. Score candidates using section weights and confidence metrics
//! 4. Select the highest-scoring candidate as the North Star goal
//!
//! # Pattern: FAIL FAST
//!
//! All invalid inputs trigger immediate assertion failures rather than
//! silent fallbacks. This ensures bugs are caught early in development.
//!
//! # Module Structure
//!
//! - `types` - Configuration and result types
//! - `constants` - Keywords and purpose starters
//! - `extraction` - Goal candidate extraction logic
//! - `scoring` - Candidate scoring algorithms
//! - `service` - Main BootstrapService implementation

mod constants;
mod extraction;
mod scoring;
mod service;
mod types;

#[cfg(test)]
mod tests;

// Re-export public API
pub use scoring::CandidateScoring;
pub use service::BootstrapService;
pub use types::{BootstrapResult, BootstrapServiceConfig, GoalCandidate};

// Re-export constants for external use if needed
pub use constants::{GOAL_KEYWORDS, PURPOSE_STARTERS};
