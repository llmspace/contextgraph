//! Edge type definitions for the Marblestone graph model.
//!
//! # Constitution Reference
//! - edge_model.attrs: type:Semantic|Temporal|Causal|Hierarchical

use serde::{Deserialize, Serialize};
use std::fmt;

/// Type of relationship between two nodes in the graph.
///
/// Each edge type represents a distinct semantic relationship with
/// different traversal and weighting characteristics:
/// - Semantic: Similarity-based connections
/// - Temporal: Time-ordered sequences
/// - Causal: Cause-effect relationships
/// - Hierarchical: Parent-child taxonomies
///
/// # Constitution Reference
/// - edge_model.attrs: type:Semantic|Temporal|Causal|Hierarchical
///
/// # Example
/// ```rust
/// use context_graph_core::marblestone::EdgeType;
///
/// let edge = EdgeType::Causal;
/// assert_eq!(edge.to_string(), "causal");
/// assert_eq!(edge.default_weight(), 0.8);
/// ```
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum EdgeType {
    /// Semantic similarity relationship.
    /// Nodes share similar meaning, topic, or conceptual space.
    Semantic,

    /// Temporal sequence relationship.
    /// Source node occurred before target node in time.
    Temporal,

    /// Causal relationship.
    /// Source node causes, influences, or triggers target node.
    Causal,

    /// Hierarchical relationship.
    /// Source node is a parent, category, or ancestor of target node.
    Hierarchical,
}

impl EdgeType {
    /// Returns a human-readable description of this edge type.
    #[inline]
    pub fn description(&self) -> &'static str {
        match self {
            Self::Semantic => "Semantic similarity - nodes share similar meaning or topic",
            Self::Temporal => "Temporal sequence - source precedes target in time",
            Self::Causal => "Causal relationship - source causes or influences target",
            Self::Hierarchical => "Hierarchical - source is parent or ancestor of target",
        }
    }

    /// Returns all edge type variants as an array.
    #[inline]
    pub fn all() -> [EdgeType; 4] {
        [
            Self::Semantic,
            Self::Temporal,
            Self::Causal,
            Self::Hierarchical,
        ]
    }

    /// Returns the default base weight for this edge type.
    ///
    /// These weights reflect the inherent reliability of each relationship type:
    /// - Semantic (0.5): Variable based on embedding similarity
    /// - Temporal (0.7): Time relationships are usually reliable
    /// - Causal (0.8): Strong evidence when established
    /// - Hierarchical (0.9): Taxonomy relationships are very strong
    #[inline]
    pub fn default_weight(&self) -> f32 {
        match self {
            Self::Semantic => 0.5,
            Self::Temporal => 0.7,
            Self::Causal => 0.8,
            Self::Hierarchical => 0.9,
        }
    }
}

impl Default for EdgeType {
    /// Returns `EdgeType::Semantic` as the default.
    /// Semantic is the most common edge type in knowledge graphs.
    #[inline]
    fn default() -> Self {
        Self::Semantic
    }
}

impl fmt::Display for EdgeType {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let s = match self {
            Self::Semantic => "semantic",
            Self::Temporal => "temporal",
            Self::Causal => "causal",
            Self::Hierarchical => "hierarchical",
        };
        write!(f, "{}", s)
    }
}
