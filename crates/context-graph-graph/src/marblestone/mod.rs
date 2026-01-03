//! Marblestone neurotransmitter integration.
//!
//! This module re-exports types from context-graph-core and adds
//! graph-specific operations for NT-weighted edge traversal.
//!
//! # Neurotransmitter Model
//!
//! Edges in the graph have NT weights that modulate effective edge weight
//! based on the current domain context:
//!
//! ```text
//! w_eff = base * (1 + excitatory - inhibitory + 0.5 * modulatory)
//! ```
//!
//! # Domain-Specific Modulation
//!
//! Different domains activate different NT profiles:
//! - Code: High modulatory for pattern matching
//! - Legal: High inhibitory for precise boundaries
//! - Medical: High excitatory for causal reasoning
//! - Creative: Balanced for exploration
//!
//! # Components
//!
//! - Re-exports from context-graph-core
//! - Domain-aware search (TODO: M04-T19)
//! - NT modulation functions (TODO: M04-T26)
//!
//! # Constitution Reference
//!
//! - edge_model.nt_weights: Definition and formula
//! - edge_model.nt_weights.domain: Code|Legal|Medical|Creative|Research|General

// Re-export from core for convenience
pub use context_graph_core::marblestone::{Domain, EdgeType, NeurotransmitterWeights};

// TODO: M04-T19 - Implement domain-aware search
// pub fn search_with_domain(
//     index: &FaissGpuIndex,
//     storage: &GraphStorage,
//     query: &[f32],
//     domain: Domain,
//     k: usize,
// ) -> GraphResult<Vec<SearchResultWithNt>>

// TODO: M04-T26 - Implement NT modulation
// pub fn compute_effective_weight(
//     base_weight: f32,
//     nt_weights: &NeurotransmitterWeights,
// ) -> f32 {
//     base_weight * (1.0 + nt_weights.excitatory - nt_weights.inhibitory + 0.5 * nt_weights.modulatory)
// }
