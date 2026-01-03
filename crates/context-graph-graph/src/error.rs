//! Error types for Knowledge Graph operations.
//!
//! This module provides comprehensive error handling for all graph operations
//! including FAISS index, hyperbolic geometry, RocksDB storage, and traversal.
//!
//! # Constitution Reference
//!
//! - AP-001: Never unwrap() in prod - all errors properly typed
//! - rules: Result<T,E> for fallible ops, thiserror for derivation
//!
//! TODO: Expand error conversions in M04-T08a

use thiserror::Error;

/// Result type alias for graph operations.
pub type GraphResult<T> = Result<T, GraphError>;

/// Comprehensive error type for all graph operations.
///
/// Each variant includes context for debugging and error recovery.
/// All errors are designed to fail fast with clear messages.
///
/// TODO: M04-T08 will add additional variants as needed
#[derive(Error, Debug)]
pub enum GraphError {
    // ========== FAISS Index Errors ==========
    /// FAISS index creation failed.
    #[error("FAISS index creation failed: {0}")]
    FaissIndexCreation(String),

    /// FAISS training failed.
    #[error("FAISS training failed: {0}")]
    FaissTrainingFailed(String),

    /// FAISS search failed.
    #[error("FAISS search failed: {0}")]
    FaissSearchFailed(String),

    /// FAISS add vectors failed.
    #[error("FAISS add failed: {0}")]
    FaissAddFailed(String),

    /// Index not trained - operations require trained index.
    #[error("Index not trained - must call train() before search/add")]
    IndexNotTrained,

    /// Insufficient training data for IVF clustering.
    #[error("Insufficient training data: need at least {required} vectors, got {provided}")]
    InsufficientTrainingData { required: usize, provided: usize },

    // ========== GPU Resource Errors ==========
    /// GPU resource allocation failed.
    #[error("GPU resource allocation failed: {0}")]
    GpuResourceAllocation(String),

    /// GPU memory transfer failed.
    #[error("GPU transfer failed: {0}")]
    GpuTransferFailed(String),

    /// GPU device not available.
    #[error("GPU device not available: {0}")]
    GpuDeviceUnavailable(String),

    // ========== Storage Errors ==========
    /// RocksDB storage error.
    #[error("Storage error: {0}")]
    Storage(String),

    /// Column family not found in RocksDB.
    #[error("Column family not found: {0}")]
    ColumnFamilyNotFound(String),

    /// Data corruption detected during deserialization.
    #[error("Corrupted data in {location}: {details}")]
    CorruptedData { location: String, details: String },

    /// Storage migration failed.
    #[error("Storage migration failed: {0}")]
    MigrationFailed(String),

    // ========== Configuration Errors ==========
    /// Invalid configuration parameter.
    #[error("Invalid configuration: {0}")]
    InvalidConfig(String),

    /// Dimension mismatch between vectors.
    #[error("Dimension mismatch: expected {expected}, got {actual}")]
    DimensionMismatch { expected: usize, actual: usize },

    // ========== Graph Structure Errors ==========
    /// Node not found in graph.
    #[error("Node not found: {0}")]
    NodeNotFound(String),

    /// Edge not found in graph.
    #[error("Edge not found: source={0}, target={1}")]
    EdgeNotFound(String, String),

    /// Duplicate node ID.
    #[error("Duplicate node ID: {0}")]
    DuplicateNode(String),

    // ========== Hyperbolic Geometry Errors ==========
    /// Invalid hyperbolic point (norm >= 1.0 or invalid).
    #[error("Invalid hyperbolic point: norm {norm} >= max_norm (must be in open ball)")]
    InvalidHyperbolicPoint { norm: f32 },

    /// Invalid curvature (must be negative).
    #[error("Invalid curvature: {0} (must be negative)")]
    InvalidCurvature(f32),

    /// Mobius operation failed.
    #[error("Mobius operation failed: {0}")]
    MobiusOperationFailed(String),

    // ========== Entailment Cone Errors ==========
    /// Invalid cone aperture.
    #[error("Invalid cone aperture: {0} (must be in (0, PI))")]
    InvalidAperture(f32),

    /// Cone axis is zero vector.
    #[error("Cone axis cannot be zero vector")]
    ZeroConeAxis,

    // ========== Traversal Errors ==========
    /// Path not found between nodes.
    #[error("No path found from {0} to {1}")]
    PathNotFound(String, String),

    /// Traversal depth exceeded.
    #[error("Traversal depth limit exceeded: {0}")]
    DepthLimitExceeded(usize),

    /// Cycle detected during traversal.
    #[error("Cycle detected at node: {0}")]
    CycleDetected(String),

    // ========== Validation Errors ==========
    /// Vector ID mismatch between index and storage.
    #[error("Vector ID mismatch: {0}")]
    VectorIdMismatch(String),

    /// Invalid NT weights (must be in [0,1]).
    #[error("Invalid NT weights: {field} = {value} (must be in [0.0, 1.0])")]
    InvalidNtWeights { field: String, value: f32 },

    // ========== Serialization Errors ==========
    /// Serialization error.
    #[error("Serialization error: {0}")]
    Serialization(String),

    /// Deserialization error.
    #[error("Deserialization error: {0}")]
    Deserialization(String),

    // ========== I/O Errors ==========
    /// IO error.
    #[error("IO error: {0}")]
    Io(#[from] std::io::Error),
}

// Error conversions will be added in M04-T08a
// impl From<rocksdb::Error> for GraphError { ... }
// impl From<bincode::Error> for GraphError { ... }

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_error_display_index_not_trained() {
        let err = GraphError::IndexNotTrained;
        let msg = err.to_string();
        assert!(msg.contains("not trained"));
        assert!(msg.contains("train()"));
    }

    #[test]
    fn test_error_display_insufficient_training_data() {
        let err = GraphError::InsufficientTrainingData {
            required: 4194304,
            provided: 1000,
        };
        let msg = err.to_string();
        assert!(msg.contains("4194304"));
        assert!(msg.contains("1000"));
    }

    #[test]
    fn test_error_display_invalid_hyperbolic_point() {
        let err = GraphError::InvalidHyperbolicPoint { norm: 1.5 };
        let msg = err.to_string();
        assert!(msg.contains("1.5"));
        assert!(msg.contains("norm"));
    }

    #[test]
    fn test_error_display_dimension_mismatch() {
        let err = GraphError::DimensionMismatch {
            expected: 1536,
            actual: 768,
        };
        let msg = err.to_string();
        assert!(msg.contains("1536"));
        assert!(msg.contains("768"));
    }

    #[test]
    fn test_error_display_node_not_found() {
        let err = GraphError::NodeNotFound("abc-123".to_string());
        let msg = err.to_string();
        assert!(msg.contains("abc-123"));
    }

    #[test]
    fn test_error_display_edge_not_found() {
        let err = GraphError::EdgeNotFound("node-a".to_string(), "node-b".to_string());
        let msg = err.to_string();
        assert!(msg.contains("node-a"));
        assert!(msg.contains("node-b"));
    }

    #[test]
    fn test_error_display_corrupted_data() {
        let err = GraphError::CorruptedData {
            location: "edges_cf".to_string(),
            details: "invalid bincode".to_string(),
        };
        let msg = err.to_string();
        assert!(msg.contains("edges_cf"));
        assert!(msg.contains("invalid bincode"));
    }

    #[test]
    fn test_error_display_invalid_nt_weights() {
        let err = GraphError::InvalidNtWeights {
            field: "excitatory".to_string(),
            value: 1.5,
        };
        let msg = err.to_string();
        assert!(msg.contains("excitatory"));
        assert!(msg.contains("1.5"));
    }

    #[test]
    fn test_graph_result_type_alias() {
        fn example_fn() -> GraphResult<u32> {
            Ok(42)
        }
        assert_eq!(example_fn().unwrap(), 42);
    }

    #[test]
    fn test_io_error_conversion() {
        let io_err = std::io::Error::new(std::io::ErrorKind::NotFound, "file not found");
        let graph_err: GraphError = io_err.into();
        assert!(matches!(graph_err, GraphError::Io(_)));
    }
}
