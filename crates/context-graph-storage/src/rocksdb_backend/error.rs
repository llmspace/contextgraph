//! Storage error types for RocksDB backend.
//!
//! These errors cover database lifecycle operations and CRUD operations.
//! Designed for fail-fast debugging with descriptive error messages.

use crate::serialization::SerializationError;
use context_graph_core::marblestone::EdgeType;
use context_graph_core::types::{NodeId, ValidationError};
use thiserror::Error;

/// Storage operation errors.
///
/// These errors cover database lifecycle operations and CRUD operations.
/// Designed for fail-fast debugging with descriptive error messages.
///
/// # TASK-M02-017 Additions
/// - `NotFound`: Node/entity not found by ID
/// - `Serialization`: Serialization/deserialization errors
/// - `ValidationFailed`: Node validation failed before storage
#[derive(Debug, Error)]
pub enum StorageError {
    /// Database failed to open.
    #[error("Failed to open database at '{path}': {message}")]
    OpenFailed { path: String, message: String },

    /// Column family not found (should never happen if DB opened correctly).
    #[error("Column family '{name}' not found")]
    ColumnFamilyNotFound { name: String },

    /// Write operation failed.
    #[error("Write failed: {0}")]
    WriteFailed(String),

    /// Read operation failed.
    #[error("Read failed: {0}")]
    ReadFailed(String),

    /// Flush operation failed.
    #[error("Flush failed: {0}")]
    FlushFailed(String),

    /// Node not found by ID.
    ///
    /// Returned by `get_node()`, `update_node()`, and `delete_node()` when
    /// the requested node does not exist in the database.
    #[error("Node not found: {id}")]
    NotFound {
        /// The node ID that was not found (as string for display)
        id: String,
    },

    /// Serialization or deserialization error.
    ///
    /// Wraps errors from the serialization module during storage operations.
    #[error("Serialization error: {0}")]
    Serialization(String),

    /// Node validation failed.
    ///
    /// Returned when `MemoryNode::validate()` fails before storage.
    /// Fail fast: invalid nodes are never stored.
    #[error("Validation error: {0}")]
    ValidationFailed(String),

    /// Index corruption detected during scan or validation.
    ///
    /// Indicates data integrity issues in secondary indexes.
    /// Should trigger investigation and potential index rebuild.
    #[error("Index corruption detected in {index_name}: {details}")]
    IndexCorrupted {
        /// Name of the corrupted index (e.g., "johari_open", "temporal")
        index_name: String,
        /// Details about the corruption (e.g., "UUID parse failed")
        details: String,
    },

    /// Generic internal error for unexpected failures.
    ///
    /// Used for internal errors that don't fit other categories.
    /// Should include diagnostic information for debugging.
    #[error("Internal storage error: {0}")]
    Internal(String),
}

impl From<SerializationError> for StorageError {
    fn from(e: SerializationError) -> Self {
        StorageError::Serialization(e.to_string())
    }
}

impl From<ValidationError> for StorageError {
    fn from(e: ValidationError) -> Self {
        StorageError::ValidationFailed(e.to_string())
    }
}

impl StorageError {
    /// Create NotFound error for a missing MemoryNode.
    ///
    /// # Arguments
    /// * `id` - The NodeId (UUID) that was not found
    ///
    /// # Example
    /// ```ignore
    /// let error = StorageError::not_found_node(node_id);
    /// ```
    pub fn not_found_node(id: NodeId) -> Self {
        StorageError::NotFound { id: id.to_string() }
    }

    /// Create NotFound error for a missing GraphEdge.
    ///
    /// Formats the edge identifier as "source->target:EdgeType" for clarity.
    ///
    /// # Arguments
    /// * `source` - The source node ID
    /// * `target` - The target node ID
    /// * `edge_type` - The type of edge
    ///
    /// # Example
    /// ```ignore
    /// let error = StorageError::not_found_edge(source_id, target_id, EdgeType::Semantic);
    /// ```
    pub fn not_found_edge(source: NodeId, target: NodeId, edge_type: EdgeType) -> Self {
        StorageError::NotFound {
            id: format!("{}->{}:{:?}", source, target, edge_type),
        }
    }

    /// Create NotFound error for a missing Embedding.
    ///
    /// Prefixes with "embedding:" to distinguish from node lookups.
    ///
    /// # Arguments
    /// * `id` - The NodeId (UUID) whose embedding was not found
    ///
    /// # Example
    /// ```ignore
    /// let error = StorageError::not_found_embedding(node_id);
    /// ```
    pub fn not_found_embedding(id: NodeId) -> Self {
        StorageError::NotFound {
            id: format!("embedding:{}", id),
        }
    }
}

/// Convenient Result type for storage operations.
///
/// All storage operations should return `StorageResult<T>` instead of
/// the more verbose `Result<T, StorageError>`.
pub type StorageResult<T> = Result<T, StorageError>;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_error_open_failed() {
        let error = StorageError::OpenFailed {
            path: "/tmp/test".to_string(),
            message: "permission denied".to_string(),
        };
        let msg = error.to_string();
        assert!(msg.contains("/tmp/test"));
        assert!(msg.contains("permission denied"));
    }

    #[test]
    fn test_error_column_family_not_found() {
        let error = StorageError::ColumnFamilyNotFound {
            name: "unknown_cf".to_string(),
        };
        let msg = error.to_string();
        assert!(msg.contains("unknown_cf"));
    }

    #[test]
    fn test_error_write_failed() {
        let error = StorageError::WriteFailed("disk full".to_string());
        assert!(error.to_string().contains("disk full"));
    }

    #[test]
    fn test_error_read_failed() {
        let error = StorageError::ReadFailed("io error".to_string());
        assert!(error.to_string().contains("io error"));
    }

    #[test]
    fn test_error_flush_failed() {
        let error = StorageError::FlushFailed("sync failed".to_string());
        assert!(error.to_string().contains("sync failed"));
    }

    #[test]
    fn test_error_debug() {
        let error = StorageError::WriteFailed("test".to_string());
        let debug = format!("{:?}", error);
        assert!(debug.contains("WriteFailed"));
    }

    #[test]
    fn test_error_not_found() {
        let error = StorageError::NotFound {
            id: "550e8400-e29b-41d4-a716-446655440000".to_string(),
        };
        let msg = error.to_string();
        assert!(msg.contains("Node not found"));
        assert!(msg.contains("550e8400"));
    }

    #[test]
    fn test_error_serialization() {
        let error = StorageError::Serialization("invalid msgpack".to_string());
        let msg = error.to_string();
        assert!(msg.contains("Serialization error"));
        assert!(msg.contains("invalid msgpack"));
    }

    #[test]
    fn test_error_validation_failed() {
        let error = StorageError::ValidationFailed("importance out of range".to_string());
        let msg = error.to_string();
        assert!(msg.contains("Validation error"));
        assert!(msg.contains("importance out of range"));
    }

    #[test]
    fn test_from_serialization_error() {
        let ser_error = SerializationError::SerializeFailed("test".to_string());
        let storage_error: StorageError = ser_error.into();
        assert!(matches!(storage_error, StorageError::Serialization(_)));
    }

    #[test]
    fn test_from_validation_error() {
        let val_error = ValidationError::InvalidEmbeddingDimension {
            expected: 1536,
            actual: 100,
        };
        let storage_error: StorageError = val_error.into();
        assert!(matches!(storage_error, StorageError::ValidationFailed(_)));
    }

    // ========== TASK-M02-025: New Variant Tests ==========

    #[test]
    fn test_error_index_corrupted() {
        let error = StorageError::IndexCorrupted {
            index_name: "johari_open".to_string(),
            details: "UUID parse failed".to_string(),
        };
        let msg = error.to_string();
        assert!(msg.contains("johari_open"));
        assert!(msg.contains("UUID parse failed"));
        assert!(msg.contains("Index corruption"));
    }

    #[test]
    fn test_error_internal() {
        let error = StorageError::Internal("unexpected state".to_string());
        let msg = error.to_string();
        assert!(msg.contains("Internal storage error"));
        assert!(msg.contains("unexpected state"));
    }

    // ========== TASK-M02-025: Convenience Constructor Tests ==========

    #[test]
    fn test_not_found_node() {
        let id = uuid::Uuid::new_v4();
        let error = StorageError::not_found_node(id);
        let msg = error.to_string();
        assert!(msg.contains(&id.to_string()));
        assert!(msg.contains("Node not found"));
    }

    #[test]
    fn test_not_found_edge() {
        let source = uuid::Uuid::new_v4();
        let target = uuid::Uuid::new_v4();
        let error = StorageError::not_found_edge(source, target, EdgeType::Semantic);
        let msg = error.to_string();
        // Check that source and target UUIDs are in the message
        assert!(msg.contains(&source.to_string()[..8]));
        assert!(msg.contains(&target.to_string()[..8]));
        assert!(msg.contains("Semantic"));
    }

    #[test]
    fn test_not_found_embedding() {
        let id = uuid::Uuid::new_v4();
        let error = StorageError::not_found_embedding(id);
        let msg = error.to_string();
        assert!(msg.contains("embedding:"));
        assert!(msg.contains(&id.to_string()));
    }

    // ========== TASK-M02-025: StorageResult Type Alias Test ==========

    #[test]
    fn test_storage_result_type_alias() {
        fn returns_ok_result() -> StorageResult<String> {
            Ok("test".to_string())
        }

        fn returns_err_result() -> StorageResult<String> {
            Err(StorageError::Internal("test error".to_string()))
        }

        assert!(returns_ok_result().is_ok());
        assert!(returns_err_result().is_err());
    }

    // ========== TASK-M02-025: Edge Case Tests ==========

    #[test]
    fn edge_case_index_corrupted_empty_details() {
        println!("=== EDGE CASE 1: Empty details string ===");
        let error = StorageError::IndexCorrupted {
            index_name: "".to_string(),
            details: "".to_string(),
        };
        println!("BEFORE: Creating error with empty strings");
        let msg = error.to_string();
        println!("AFTER: message = '{}'", msg);
        assert!(msg.contains("Index corruption"));
        println!("RESULT: PASS - Handles empty strings");
    }

    #[test]
    fn edge_case_unicode_in_internal_error() {
        println!("=== EDGE CASE 2: Unicode in Internal error ===");
        let error = StorageError::Internal("错误: 日本語 🔥".to_string());
        println!("BEFORE: Creating error with unicode");
        let msg = error.to_string();
        println!("AFTER: message = '{}'", msg);
        assert!(msg.contains("错误"));
        assert!(msg.contains("🔥"));
        println!("RESULT: PASS - Unicode preserved");
    }

    #[test]
    fn edge_case_nil_uuid_in_not_found() {
        println!("=== EDGE CASE 3: Nil UUID in not_found_node ===");
        let nil = uuid::Uuid::nil();
        println!("BEFORE: Creating not_found_node with nil UUID");
        let error = StorageError::not_found_node(nil);
        let msg = error.to_string();
        println!("AFTER: error = '{}'", msg);
        assert!(msg.contains("00000000-0000-0000-0000-000000000000"));
        println!("RESULT: PASS - Nil UUID formatted correctly");
    }
}
