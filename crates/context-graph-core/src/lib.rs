//! Context Graph Core Library
//!
//! Provides core domain types, traits, and stub implementations for the
//! Ultimate Context Graph system implementing UTL (Unified Theory of Learning).
//!
//! # Architecture
//!
//! This crate defines:
//! - Domain types (`MemoryNode`, `GraphEdge`, `JohariQuadrant`, etc.)
//! - Core traits (`UTLProcessor`, `MemoryStore`, `NervousLayer`, etc.)
//! - Error types and result aliases
//! - Configuration structures
//!
//! # Feature Flags
//!
//! - `real-utl`: Enables the real UTL processor adapter (bridges to context-graph-utl)
//!
//! # Example
//!
//! ```
//! use context_graph_core::types::MemoryNode;
//! use context_graph_core::traits::{MemoryStore, SearchOptions};
//!
//! // Create search options for querying
//! let options = SearchOptions::new(10)
//!     .with_min_similarity(0.8);
//! assert_eq!(options.top_k, 10);
//! ```

pub mod config;
pub mod error;
pub mod marblestone;
pub mod stubs;
pub mod traits;
pub mod types;

// Adapters module - only available with real-utl feature
#[cfg(feature = "real-utl")]
pub mod adapters;

// Re-exports for convenience
pub use config::Config;
pub use error::{CoreError, CoreResult};
pub use marblestone::{Domain, EdgeType, NeurotransmitterWeights};

// Re-export UtlProcessorAdapter when real-utl feature is enabled
#[cfg(feature = "real-utl")]
pub use adapters::UtlProcessorAdapter;
