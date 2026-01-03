//! Knowledge Graph with FAISS GPU Vector Search and Hyperbolic Geometry
//!
//! This crate provides the Knowledge Graph layer for the Context Graph system,
//! combining FAISS GPU-accelerated vector similarity search with hyperbolic
//! geometry for hierarchical reasoning.
//!
//! # Architecture
//!
//! - **config**: Index, hyperbolic, and cone configuration types
//! - **error**: Comprehensive error handling with GraphError
//! - **hyperbolic**: Poincare ball model with Mobius operations
//! - **entailment**: Entailment cones for O(1) IS-A queries
//! - **index**: FAISS GPU IVF-PQ index wrapper
//! - **storage**: RocksDB backend for graph persistence
//! - **traversal**: BFS, DFS, and A* graph traversal
//! - **marblestone**: Marblestone NT integration
//! - **query**: High-level query operations
//!
//! # Constitution Reference
//!
//! - TECH-GRAPH-004: Technical specification
//! - edge_model.nt_weights: Neurotransmitter weighting
//! - perf.latency.faiss_1M_k100: <2ms target
//!
//! # Example
//!
//! ```
//! use context_graph_graph::config::IndexConfig;
//! use context_graph_graph::error::GraphResult;
//!
//! fn example() -> GraphResult<()> {
//!     let config = IndexConfig::default();
//!     assert_eq!(config.dimension, 1536);
//!     Ok(())
//! }
//! ```

pub mod config;
pub mod entailment;
pub mod error;
pub mod hyperbolic;
pub mod index;
pub mod marblestone;
pub mod query;
pub mod storage;
pub mod traversal;

// Re-exports for convenience
pub use config::{ConeConfig, HyperbolicConfig, IndexConfig};
pub use error::{GraphError, GraphResult};

// Re-export core types for convenience
pub use context_graph_core::marblestone::{Domain, EdgeType, NeurotransmitterWeights};
pub use context_graph_core::types::{EmbeddingVector, NodeId, DEFAULT_EMBEDDING_DIM};
