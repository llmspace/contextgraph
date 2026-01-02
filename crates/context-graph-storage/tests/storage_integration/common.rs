//! Common test utilities for integration tests.
//!
//! Provides shared helpers for creating test nodes, edges, and database instances.

use context_graph_core::marblestone::{Domain, EdgeType};
use context_graph_core::types::{EmbeddingVector, GraphEdge, MemoryNode, NodeId};
use context_graph_storage::RocksDbMemex;
use tempfile::TempDir;

/// Create a valid normalized embedding (magnitude ~1.0)
pub fn create_valid_embedding() -> EmbeddingVector {
    const DIM: usize = 1536;
    let val = 1.0_f32 / (DIM as f32).sqrt();
    vec![val; DIM]
}

/// Create a test node with default content
pub fn create_test_node() -> MemoryNode {
    MemoryNode::new("Test content".to_string(), create_valid_embedding())
}

/// Create a test node with custom content
pub fn create_node_with_content(content: &str) -> MemoryNode {
    MemoryNode::new(content.to_string(), create_valid_embedding())
}

/// Create a test edge with default settings
pub fn create_test_edge(source: NodeId, target: NodeId) -> GraphEdge {
    GraphEdge::new(source, target, EdgeType::Semantic, Domain::General)
}

/// Setup a fresh RocksDB instance in a temporary directory
pub fn setup_db() -> (RocksDbMemex, TempDir) {
    let tmp = TempDir::new().expect("create temp dir");
    let db = RocksDbMemex::open(tmp.path()).expect("open db");
    (db, tmp)
}
