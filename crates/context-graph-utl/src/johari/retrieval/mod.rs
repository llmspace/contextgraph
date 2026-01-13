//! Quadrant-aware retrieval strategies and suggested actions.
//!
//! This module provides retrieval weighting and action recommendations based on
//! Johari quadrant classification, implementing the UTL constitution specifications
//! for memory retrieval behavior.
//!
//! # Retrieval Strategies
//!
//! Each quadrant has different retrieval characteristics:
//! - **Open**: Direct recall with full weight (1.0)
//! - **Blind**: Discovery-focused with high weight (configurable)
//! - **Hidden**: Private with reduced weight (configurable)
//! - **Unknown**: Frontier exploration with medium weight (configurable)
//!
//! # Suggested Actions
//!
//! Based on the constitution.yaml utl.johari specifications (lines 154-157):
//! - **Open** (ΔS<0.5, ΔC>0.5) -> DirectRecall
//! - **Blind** (ΔS>0.5, ΔC<0.5) -> TriggerDream (sleep consolidation)
//! - **Hidden** (ΔS<0.5, ΔC<0.5) -> GetNeighborhood (context exploration)
//! - **Unknown** (ΔS>0.5, ΔC>0.5) -> EpistemicAction (belief update)

mod action;
mod functions;
mod quadrant_retrieval;

#[cfg(test)]
mod tests;

// Re-export all public API for backwards compatibility
pub use action::SuggestedAction;
pub use functions::{get_retrieval_weight, get_suggested_action};
pub use quadrant_retrieval::QuadrantRetrieval;
