//! Storage error types for RocksDB backend.
//!
//! This module defines the error types for all RocksDB storage operations.
//! Errors are designed for fail-fast debugging with descriptive messages.
//!
//! # Error Categories
//!
//! - **Lifecycle Errors**: `OpenFailed`, `FlushFailed` - database open/close operations
//! - **CRUD Errors**: `WriteFailed`, `ReadFailed`, `NotFound` - data operations
//! - **Data Integrity**: `Serialization`, `ValidationFailed`, `IndexCorrupted`
//! - **Internal**: `ColumnFamilyNotFound`, `Internal` - unexpected failures
//!
//! # Constitution Reference
//!
//! - SEC-06: Soft delete 30-day recovery (see `delete_node` documentation)
//! - AP-001: Never unwrap() in prod - all errors bubble up via `StorageResult<T>`
//!
//! # Example: Error Handling Pattern
//!
//! ```rust
//! use context_graph_storage::{RocksDbMemex, StorageError, Memex};
//! use tempfile::TempDir;
//! use uuid::Uuid;
//!
//! let tmp = TempDir::new().unwrap();
//! let memex = RocksDbMemex::open(tmp.path()).unwrap();
//!
//! // Attempting to get a non-existent node
//! let missing_id = Uuid::new_v4();
//! match memex.get_node(&missing_id) {
//!     Ok(node) => println!("Found: {}", node.content),
//!     Err(StorageError::NotFound { id }) => println!("Node {} not found", id),
//!     Err(e) => println!("Other error: {}", e),
//! }
//! ```

use crate::serialization::SerializationError;
use context_graph_core::marblestone::EdgeType;
use context_graph_core::types::{NodeId, ValidationError};
use thiserror::Error;

/// Storage operation errors for RocksDB backend.
///
/// Provides typed errors for all storage operations with descriptive messages.
/// Implements `std::error::Error` and `Display` via `thiserror`.
///
/// # Design Philosophy
///
/// - **Fail-Fast**: Errors contain enough context for immediate debugging
/// - **Type Safety**: Each error variant has specific fields for its context
/// - **Composable**: `From` implementations allow `?` operator usage
///
/// # Error Matching
///
/// ```rust
/// use context_graph_storage::StorageError;
///
/// fn handle_error(err: StorageError) -> &'static str {
///     match err {
///         StorageError::NotFound { .. } => "Entity not found - check ID",
///         StorageError::ValidationFailed(_) => "Invalid data - check input",
///         StorageError::OpenFailed { .. } => "DB unavailable - check path/permissions",
///         StorageError::WriteFailed(_) => "Write failed - check disk space",
///         StorageError::ReadFailed(_) => "Read failed - check DB health",
///         StorageError::FlushFailed(_) => "Flush failed - check disk",
///         StorageError::Serialization(_) => "Data corruption - investigate",
///         StorageError::ColumnFamilyNotFound { .. } => "Internal error - report bug",
///         StorageError::IndexCorrupted { .. } => "Index corruption - rebuild needed",
///         StorageError::Internal(_) => "Internal error - report bug",
///     }
/// }
/// ```
#[derive(Debug, Error)]
pub enum StorageError {
    /// Database failed to open at the specified path.
    ///
    /// # When This Occurs
    ///
    /// - Path does not exist and `create_if_missing` is false
    /// - Insufficient permissions to read/write the directory
    /// - Database files are corrupted or locked by another process
    /// - Disk is full or has I/O errors
    ///
    /// # Example
    ///
    /// ```rust
    /// use context_graph_storage::StorageError;
    ///
    /// let error = StorageError::OpenFailed {
    ///     path: "/invalid/path".to_string(),
    ///     message: "No such file or directory".to_string(),
    /// };
    /// assert!(error.to_string().contains("/invalid/path"));
    /// ```
    #[error("Failed to open database at '{path}': {message}")]
    OpenFailed {
        /// The path where database open was attempted
        path: String,
        /// The underlying error message from RocksDB
        message: String,
    },

    /// Column family not found in the database.
    ///
    /// # When This Occurs
    ///
    /// This error should never occur in normal operation. It indicates:
    /// - Database was opened without all required column families
    /// - Database schema mismatch between versions
    /// - Internal bug in column family initialization
    ///
    /// # Example
    ///
    /// ```rust
    /// use context_graph_storage::StorageError;
    ///
    /// let error = StorageError::ColumnFamilyNotFound {
    ///     name: "unknown_cf".to_string(),
    /// };
    /// assert!(error.to_string().contains("unknown_cf"));
    /// ```
    #[error("Column family '{name}' not found")]
    ColumnFamilyNotFound {
        /// Name of the missing column family
        name: String,
    },

    /// Write operation failed.
    ///
    /// # When This Occurs
    ///
    /// - Disk is full or has I/O errors
    /// - Database is read-only
    /// - WAL (Write-Ahead Log) write failed
    /// - Memory pressure causing write buffer issues
    ///
    /// # Example
    ///
    /// ```rust
    /// use context_graph_storage::StorageError;
    ///
    /// let error = StorageError::WriteFailed("Disk full".to_string());
    /// assert!(error.to_string().contains("Disk full"));
    /// ```
    #[error("Write failed: {0}")]
    WriteFailed(String),

    /// Read operation failed.
    ///
    /// # When This Occurs
    ///
    /// - Database file corruption
    /// - I/O error during read
    /// - Memory allocation failure
    /// - SST file read error
    ///
    /// # Example
    ///
    /// ```rust
    /// use context_graph_storage::StorageError;
    ///
    /// let error = StorageError::ReadFailed("I/O error".to_string());
    /// assert!(error.to_string().contains("I/O error"));
    /// ```
    #[error("Read failed: {0}")]
    ReadFailed(String),

    /// Flush operation failed.
    ///
    /// # When This Occurs
    ///
    /// - Disk is full
    /// - I/O error during flush
    /// - Database shutdown during flush
    ///
    /// # Example
    ///
    /// ```rust
    /// use context_graph_storage::StorageError;
    ///
    /// let error = StorageError::FlushFailed("Sync failed".to_string());
    /// assert!(error.to_string().contains("Sync failed"));
    /// ```
    #[error("Flush failed: {0}")]
    FlushFailed(String),

    /// Entity not found by ID.
    ///
    /// # When This Occurs
    ///
    /// - `get_node()`: Node with given ID does not exist
    /// - `update_node()`: Attempting to update non-existent node
    /// - `delete_node()`: Attempting to delete non-existent node
    /// - `get_edge()`: Edge with given composite key does not exist
    /// - `get_embedding()`: No embedding stored for given node ID
    ///
    /// # ID Format
    ///
    /// The `id` field contains a formatted identifier:
    /// - Nodes: UUID string (e.g., "550e8400-e29b-41d4-a716-446655440000")
    /// - Edges: "source->target:EdgeType" format
    /// - Embeddings: "embedding:UUID" format
    ///
    /// # Example
    ///
    /// ```rust
    /// use context_graph_storage::StorageError;
    /// use uuid::Uuid;
    ///
    /// // Node not found
    /// let node_id = Uuid::new_v4();
    /// let error = StorageError::not_found_node(node_id);
    /// assert!(error.to_string().contains("Node not found"));
    ///
    /// // Embedding not found
    /// let error = StorageError::not_found_embedding(node_id);
    /// assert!(error.to_string().contains("embedding:"));
    /// ```
    #[error("Node not found: {id}")]
    NotFound {
        /// The entity ID that was not found (formatted as string)
        id: String,
    },

    /// Serialization or deserialization failed.
    ///
    /// # When This Occurs
    ///
    /// - Corrupted data in database
    /// - Schema mismatch between stored and expected format
    /// - Invalid MessagePack or bincode data
    /// - Truncated data (incomplete write)
    ///
    /// # Example
    ///
    /// ```rust
    /// use context_graph_storage::StorageError;
    ///
    /// let error = StorageError::Serialization("Invalid msgpack".to_string());
    /// assert!(error.to_string().contains("Serialization error"));
    /// ```
    #[error("Serialization error: {0}")]
    Serialization(String),

    /// Node validation failed before storage.
    ///
    /// # When This Occurs
    ///
    /// Returned when `MemoryNode::validate()` fails. Common causes:
    /// - Embedding dimension mismatch (expected 1536, got different)
    /// - Embedding not normalized (magnitude not ~1.0)
    /// - Content exceeds MAX_CONTENT_SIZE (1MB)
    /// - Importance out of range [0.0, 1.0]
    /// - Emotional valence out of range [-1.0, 1.0]
    ///
    /// # Fail-Fast Principle
    ///
    /// Invalid nodes are NEVER stored. This prevents data corruption
    /// and ensures all persisted data meets invariants.
    ///
    /// # Example
    ///
    /// ```rust
    /// use context_graph_storage::StorageError;
    ///
    /// let error = StorageError::ValidationFailed(
    ///     "Importance 1.5 out of range [0.0, 1.0]".to_string()
    /// );
    /// assert!(error.to_string().contains("Validation error"));
    /// ```
    #[error("Validation error: {0}")]
    ValidationFailed(String),

    /// Index corruption detected during scan or validation.
    ///
    /// # When This Occurs
    ///
    /// - Secondary index contains invalid UUID bytes
    /// - Index references non-existent primary data
    /// - Partial write left index in inconsistent state
    /// - Manual database manipulation corrupted indexes
    ///
    /// # Recovery
    ///
    /// When this error occurs:
    /// 1. Log the error with full details
    /// 2. Consider rebuilding the affected index
    /// 3. Investigate root cause (disk errors, bugs, etc.)
    ///
    /// # Example
    ///
    /// ```rust
    /// use context_graph_storage::StorageError;
    ///
    /// let error = StorageError::IndexCorrupted {
    ///     index_name: "temporal".to_string(),
    ///     details: "UUID parse failed for key".to_string(),
    /// };
    /// assert!(error.to_string().contains("temporal"));
    /// assert!(error.to_string().contains("UUID parse failed"));
    /// ```
    #[error("Index corruption detected in {index_name}: {details}")]
    IndexCorrupted {
        /// Name of the corrupted index (e.g., "temporal", "tags", "sources")
        index_name: String,
        /// Details about what corruption was detected
        details: String,
    },

    /// Generic internal error for unexpected failures.
    ///
    /// # When This Occurs
    ///
    /// Used for errors that don't fit other categories:
    /// - Unexpected RocksDB internal errors
    /// - Logic errors that should never happen
    /// - Edge cases not covered by specific variants
    ///
    /// # Debugging
    ///
    /// The message should contain enough context for debugging.
    /// If you see this error frequently, consider adding a
    /// specific variant for that error case.
    ///
    /// # Example
    ///
    /// ```rust
    /// use context_graph_storage::StorageError;
    ///
    /// let error = StorageError::Internal("Unexpected state: X".to_string());
    /// assert!(error.to_string().contains("Internal storage error"));
    /// ```
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

impl From<rocksdb::Error> for StorageError {
    fn from(e: rocksdb::Error) -> Self {
        StorageError::Internal(e.to_string())
    }
}

impl StorageError {
    /// Creates a `NotFound` error for a missing `MemoryNode`.
    ///
    /// Convenience constructor that formats the UUID as a string
    /// for the error message.
    ///
    /// # Arguments
    ///
    /// * `id` - The `NodeId` (UUID) that was not found
    ///
    /// # Returns
    ///
    /// A `StorageError::NotFound` variant with the formatted node ID.
    ///
    /// # Example
    ///
    /// ```rust
    /// use context_graph_storage::StorageError;
    /// use uuid::Uuid;
    ///
    /// let node_id = Uuid::new_v4();
    /// let error = StorageError::not_found_node(node_id);
    ///
    /// assert!(error.to_string().contains("Node not found"));
    /// assert!(error.to_string().contains(&node_id.to_string()));
    /// ```
    pub fn not_found_node(id: NodeId) -> Self {
        StorageError::NotFound { id: id.to_string() }
    }

    /// Creates a `NotFound` error for a missing `GraphEdge`.
    ///
    /// Formats the edge identifier as "source->target:EdgeType" for clarity,
    /// making it easy to identify which edge was missing.
    ///
    /// # Arguments
    ///
    /// * `source` - The source node ID
    /// * `target` - The target node ID
    /// * `edge_type` - The type of edge (Semantic, Temporal, Causal, etc.)
    ///
    /// # Returns
    ///
    /// A `StorageError::NotFound` variant with the formatted edge key.
    ///
    /// # Example
    ///
    /// ```rust
    /// use context_graph_storage::StorageError;
    /// use context_graph_core::marblestone::EdgeType;
    /// use uuid::Uuid;
    ///
    /// let source = Uuid::new_v4();
    /// let target = Uuid::new_v4();
    /// let error = StorageError::not_found_edge(source, target, EdgeType::Semantic);
    ///
    /// let msg = error.to_string();
    /// assert!(msg.contains("->"));
    /// assert!(msg.contains("Semantic"));
    /// ```
    pub fn not_found_edge(source: NodeId, target: NodeId, edge_type: EdgeType) -> Self {
        StorageError::NotFound {
            id: format!("{}->{}:{:?}", source, target, edge_type),
        }
    }

    /// Creates a `NotFound` error for a missing embedding.
    ///
    /// Prefixes the ID with "embedding:" to distinguish embedding lookups
    /// from node lookups in error messages.
    ///
    /// # Arguments
    ///
    /// * `id` - The `NodeId` (UUID) whose embedding was not found
    ///
    /// # Returns
    ///
    /// A `StorageError::NotFound` variant with "embedding:" prefix.
    ///
    /// # Example
    ///
    /// ```rust
    /// use context_graph_storage::StorageError;
    /// use uuid::Uuid;
    ///
    /// let node_id = Uuid::new_v4();
    /// let error = StorageError::not_found_embedding(node_id);
    ///
    /// let msg = error.to_string();
    /// assert!(msg.contains("embedding:"));
    /// assert!(msg.contains(&node_id.to_string()));
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
            index_name: "temporal".to_string(),
            details: "UUID parse failed".to_string(),
        };
        let msg = error.to_string();
        assert!(msg.contains("temporal"));
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
