//! Goal hierarchy types for teleological alignment.
//!
//! Provides the goal tree structure that defines Strategic goals and sub-goals
//! for purpose vector computation.
//!
//! # Architecture (constitution.yaml)
//!
//! - **ARCH-02**: Goals use TeleologicalArray for apples-to-apples comparison
//! - **ARCH-03**: Goals are discovered autonomously, not manually created
//! - **ARCH-05**: All 13 embedders must be present in teleological_array
//!
//! # Example
//!
//! ```ignore
//! use context_graph_core::purpose::{GoalNode, GoalLevel, DiscoveryMethod, GoalDiscoveryMetadata};
//! use context_graph_core::types::fingerprint::SemanticFingerprint;
//!
//! // Goals are created from clustering analysis
//! let discovery = GoalDiscoveryMetadata::new(
//!     DiscoveryMethod::Clustering,
//!     0.85,  // confidence
//!     42,    // cluster_size
//!     0.78,  // coherence
//! ).unwrap();
//!
//! let centroid_fingerprint = SemanticFingerprint::zeroed();
//! let goal = GoalNode::autonomous_goal(
//!     "Emergent ML mastery goal".to_string(),
//!     GoalLevel::Strategic,
//!     centroid_fingerprint,
//!     discovery,
//! ).unwrap();
//! ```

mod discovery;
mod error;
mod hierarchy;
mod level;
mod node;

#[cfg(test)]
mod tests;

// Re-export all public types for backwards compatibility
pub use self::discovery::{DiscoveryMethod, GoalDiscoveryMetadata};
pub use self::error::{GoalHierarchyError, GoalNodeError};
pub use self::hierarchy::GoalHierarchy;
pub use self::level::GoalLevel;
pub use self::node::GoalNode;
