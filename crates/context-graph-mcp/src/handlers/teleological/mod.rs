//! Teleological MCP Tool Handlers (TELEO-H1 through TELEO-H5).
//!
//! Implements handlers for the 5 teleological tools:
//! - search_teleological: Cross-correlation search across all 13 embedders
//! - compute_teleological_vector: Compute full teleological vector from content
//! - fuse_embeddings: Fuse multiple embeddings using synergy matrix
//! - update_synergy_matrix: Adaptive learning from retrieval feedback
//! - manage_teleological_profile: CRUD for task-specific profiles
//!
//! # Module Structure
//!
//! - `types`: Parameter structs and JSON representations
//! - `conversions`: Type conversions and parsing utilities
//! - `utils`: Embedding extraction and alignment computation
//! - `helpers`: Shared helper functions for handlers
//! - `search`: TELEO-H1 search_teleological handler
//! - `compute`: TELEO-H2 compute_teleological_vector handler
//! - `fusion`: TELEO-H3 fuse_embeddings handler
//! - `feedback`: TELEO-H4 update_synergy_matrix handler
//! - `profile`: TELEO-H5 manage_teleological_profile handler

mod compute;
mod conversions;
mod feedback;
mod fusion;
mod helpers;
mod profile;
mod search;
mod types;
mod utils;

// Re-export types for external use

// Re-export conversion utilities

// Re-export utility functions
