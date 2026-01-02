//! FusedEmbedding: Final 1536D output from FuseMoE fusion.
//!
//! This module provides the primary embedding representation used throughout
//! the system for similarity search, clustering, and downstream tasks.
//!
//! # Architecture
//!
//! The FuseMoE fusion layer takes the 8320D concatenated embeddings from all
//! 12 models and produces a unified 1536D vector through sparse expert routing:
//!
//! - 8 expert networks, each specialized for different aspects
//! - Top-K routing selects the best K experts per input (K=4 per constitution.yaml)
//! - Expert weights sum to 1.0 for normalized contribution
//!
//! # Binary Format
//!
//! Core embedding serializes to exactly 6200 bytes:
//! - 6144 bytes: 1536 x f32 vector
//! - 32 bytes: 8 x f32 expert weights
//! - 4 bytes: 4 x u8 selected experts (TOP_K_EXPERTS=4)
//! - 8 bytes: u64 pipeline latency
//! - 8 bytes: u64 content hash
//! - 4 bytes: u32 aux_data length (0 if None)

// Submodules
pub mod constants;
mod core;
mod gpu;
mod operations;
mod serialization;
mod validation;

// Test modules
#[cfg(test)]
mod tests;
#[cfg(test)]
mod tests_auxiliary;
#[cfg(test)]
mod tests_operations;
#[cfg(test)]
mod tests_serialization;

// Re-export all public types for backwards compatibility
pub use core::{AuxiliaryEmbeddingData, FusedEmbedding};
