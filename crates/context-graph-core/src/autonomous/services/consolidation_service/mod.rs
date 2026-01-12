//! NORTH-013: Memory Consolidation Service
//!
//! This service merges similar memories to reduce redundancy while preserving
//! alignment with the North Star goal. It uses cosine similarity for content
//! comparison and respects configurable thresholds for merging decisions.
//!
//! # Module Structure
//!
//! - `types` - Type definitions for memory content and consolidation candidates
//! - `service` - Main consolidation service implementation

mod service;
mod types;

#[cfg(test)]
mod tests;

// Re-export all public types to maintain backwards compatibility
pub use service::ConsolidationService;
pub use types::{MemoryContent, MemoryPair, ServiceConsolidationCandidate};
