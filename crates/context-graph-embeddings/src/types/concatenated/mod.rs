//! Aggregated embedding from all 12 models for FuseMoE input.
//!
//! The `ConcatenatedEmbedding` struct collects individual `ModelEmbedding` outputs
//! and concatenates them into a single 8320-dimensional vector for FuseMoE processing.
//!
//! # Pipeline Position
//!
//! ```text
//! Individual Models (E1-E12)
//!          ↓
//!     ModelEmbedding (per model)
//!          ↓
//!     ConcatenatedEmbedding (this module) ← collects all 12
//!          ↓
//!     FuseMoE (8320D → 1536D)
//!          ↓
//!     FusedEmbedding (final output)
//! ```
//!
//! # Module Structure
//!
//! - `core`: Core struct definition and basic operations (new, set, get, etc.)
//! - `operations`: Concatenation, validation, hashing, and slicing operations
//!
//! # Example
//!
//! ```rust,ignore
//! use context_graph_embeddings::types::{ConcatenatedEmbedding, ModelEmbedding, ModelId};
//!
//! let mut concat = ConcatenatedEmbedding::new();
//!
//! // Add embeddings from each model
//! for model_id in ModelId::all() {
//!     let dim = model_id.projected_dimension();
//!     let mut emb = ModelEmbedding::new(*model_id, vec![0.1; dim], 100);
//!     emb.set_projected(true);
//!     concat.set(emb);
//! }
//!
//! // Now build the concatenated vector
//! concat.concatenate();
//! assert_eq!(concat.concatenated.len(), 8320);
//! ```

mod core;
mod operations;

#[cfg(test)]
mod tests;

// Re-export the main struct for backwards compatibility
pub use self::core::ConcatenatedEmbedding;
