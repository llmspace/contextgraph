//! Core TeleologicalMemoryStore trait for 5-stage teleological retrieval.
//!
//! This module defines the core storage trait for the Context Graph system's
//! teleological memory architecture.

use std::path::{Path, PathBuf};

use async_trait::async_trait;
use uuid::Uuid;

use crate::error::CoreResult;
use crate::types::fingerprint::{
    PurposeVector, SemanticFingerprint, SparseVector, TeleologicalFingerprint,
};

use super::backend::TeleologicalStorageBackend;
use super::options::TeleologicalSearchOptions;
use super::result::TeleologicalSearchResult;

/// Core trait for teleological memory storage operations.
///
/// This trait defines the complete interface for storing, retrieving,
/// and searching TeleologicalFingerprints. Implementations must support:
/// - Full CRUD operations with soft/hard delete
/// - Multi-space semantic search
/// - Purpose vector alignment search
/// - Sparse (SPLADE) search for efficient recall
/// - Batch operations for throughput
/// - Persistence and recovery
///
/// # Implementation Notes
///
/// - All methods are async for I/O flexibility
/// - All errors use `CoreError` variants for consistent handling
/// - The trait requires `Send + Sync` for concurrent access
/// - Implementations should log errors via `tracing` before returning
///
/// # Example
///
/// ```ignore
/// use context_graph_core::traits::TeleologicalMemoryStore;
/// use context_graph_core::stubs::InMemoryTeleologicalStore;
///
/// let store = InMemoryTeleologicalStore::new();
/// let id = store.store(fingerprint).await?;
/// let retrieved = store.retrieve(id).await?;
/// ```
#[async_trait]
pub trait TeleologicalMemoryStore: Send + Sync {
    // ==================== CRUD Operations ====================

    /// Store a new teleological fingerprint.
    ///
    /// # Arguments
    /// * `fingerprint` - The fingerprint to store
    ///
    /// # Returns
    /// The UUID assigned to the stored fingerprint.
    ///
    /// # Errors
    /// - `CoreError::StorageError` - Storage backend failure
    /// - `CoreError::ValidationError` - Invalid fingerprint data
    /// - `CoreError::SerializationError` - Serialization failure
    async fn store(&self, fingerprint: TeleologicalFingerprint) -> CoreResult<Uuid>;

    /// Retrieve a fingerprint by its UUID.
    ///
    /// # Arguments
    /// * `id` - The UUID of the fingerprint to retrieve
    ///
    /// # Returns
    /// `Some(fingerprint)` if found, `None` if not found or soft-deleted.
    ///
    /// # Errors
    /// - `CoreError::StorageError` - Storage backend failure
    /// - `CoreError::SerializationError` - Deserialization failure
    async fn retrieve(&self, id: Uuid) -> CoreResult<Option<TeleologicalFingerprint>>;

    /// Update an existing fingerprint.
    ///
    /// Replaces the entire fingerprint with the new data.
    /// The fingerprint's `id` field determines which record to update.
    ///
    /// # Arguments
    /// * `fingerprint` - The updated fingerprint (must have existing ID)
    ///
    /// # Returns
    /// `true` if updated, `false` if ID not found.
    ///
    /// # Errors
    /// - `CoreError::StorageError` - Storage backend failure
    /// - `CoreError::ValidationError` - Invalid fingerprint data
    async fn update(&self, fingerprint: TeleologicalFingerprint) -> CoreResult<bool>;

    /// Delete a fingerprint.
    ///
    /// # Arguments
    /// * `id` - The UUID of the fingerprint to delete
    /// * `soft` - If true, mark as deleted but retain data; if false, permanently remove
    ///
    /// # Returns
    /// `true` if deleted, `false` if ID not found.
    ///
    /// # Errors
    /// - `CoreError::StorageError` - Storage backend failure
    async fn delete(&self, id: Uuid, soft: bool) -> CoreResult<bool>;

    // ==================== Search Operations ====================

    /// Search by semantic similarity using the 13-embedding fingerprint.
    ///
    /// Computes similarity across all 13 embedding spaces and aggregates
    /// using Reciprocal Rank Fusion (RRF) or weighted averaging.
    ///
    /// # Arguments
    /// * `query` - The semantic fingerprint to search for
    /// * `options` - Search options (top_k, filters, etc.)
    ///
    /// # Returns
    /// Vector of search results sorted by similarity (descending).
    ///
    /// # Errors
    /// - `CoreError::StorageError` - Storage backend failure
    /// - `CoreError::ValidationError` - Invalid query fingerprint
    async fn search_semantic(
        &self,
        query: &SemanticFingerprint,
        options: TeleologicalSearchOptions,
    ) -> CoreResult<Vec<TeleologicalSearchResult>>;

    /// Search by purpose vector alignment.
    ///
    /// Finds fingerprints with similar purpose alignment to Strategic goals.
    /// Used in Stage 5 (Teleological) of the retrieval pipeline.
    ///
    /// # Arguments
    /// * `query` - The purpose vector to match
    /// * `options` - Search options (top_k, filters, etc.)
    ///
    /// # Returns
    /// Vector of search results sorted by purpose alignment (descending).
    ///
    /// # Errors
    /// - `CoreError::StorageError` - Storage backend failure
    async fn search_purpose(
        &self,
        query: &PurposeVector,
        options: TeleologicalSearchOptions,
    ) -> CoreResult<Vec<TeleologicalSearchResult>>;

    /// Full-text search using text query (generates embeddings internally).
    ///
    /// This method handles embedding generation for the text query and
    /// delegates to `search_semantic`. Implementations may cache embeddings.
    ///
    /// # Arguments
    /// * `text` - The text query to search for
    /// * `options` - Search options (top_k, filters, etc.)
    ///
    /// # Returns
    /// Vector of search results sorted by relevance (descending).
    ///
    /// # Errors
    /// - `CoreError::Embedding` - Embedding generation failure
    /// - `CoreError::StorageError` - Storage backend failure
    async fn search_text(
        &self,
        text: &str,
        options: TeleologicalSearchOptions,
    ) -> CoreResult<Vec<TeleologicalSearchResult>>;

    /// Sparse search using E13 SPLADE embeddings.
    ///
    /// Stage 1 (Recall) of the 5-stage pipeline. Uses inverted index
    /// for efficient initial candidate retrieval.
    ///
    /// # Arguments
    /// * `sparse_query` - The sparse vector query (E13 SPLADE)
    /// * `top_k` - Maximum number of candidates to return
    ///
    /// # Returns
    /// Vector of (UUID, score) pairs sorted by sparse dot product (descending).
    ///
    /// # Errors
    /// - `CoreError::StorageError` - Storage backend failure
    /// - `CoreError::IndexError` - Inverted index failure
    async fn search_sparse(
        &self,
        sparse_query: &SparseVector,
        top_k: usize,
    ) -> CoreResult<Vec<(Uuid, f32)>>;

    // ==================== Batch Operations ====================

    /// Store multiple fingerprints in a batch.
    ///
    /// More efficient than individual `store` calls for bulk ingestion.
    ///
    /// # Arguments
    /// * `fingerprints` - Vector of fingerprints to store
    ///
    /// # Returns
    /// Vector of UUIDs assigned to each fingerprint (same order as input).
    ///
    /// # Errors
    /// - `CoreError::StorageError` - Storage backend failure
    /// - `CoreError::ValidationError` - Invalid fingerprint in batch
    async fn store_batch(
        &self,
        fingerprints: Vec<TeleologicalFingerprint>,
    ) -> CoreResult<Vec<Uuid>>;

    /// Retrieve multiple fingerprints by their UUIDs.
    ///
    /// # Arguments
    /// * `ids` - Slice of UUIDs to retrieve
    ///
    /// # Returns
    /// Vector of `Option<TeleologicalFingerprint>` (same order as input).
    /// `None` entries indicate IDs not found or soft-deleted.
    ///
    /// # Errors
    /// - `CoreError::StorageError` - Storage backend failure
    async fn retrieve_batch(
        &self,
        ids: &[Uuid],
    ) -> CoreResult<Vec<Option<TeleologicalFingerprint>>>;

    // ==================== Statistics ====================

    /// Get the total number of stored fingerprints.
    ///
    /// # Returns
    /// Count of all fingerprints (excludes soft-deleted by default).
    ///
    /// # Errors
    /// - `CoreError::StorageError` - Storage backend failure
    async fn count(&self) -> CoreResult<usize>;

    /// Get total storage size in bytes.
    ///
    /// Returns the approximate heap memory used by the store.
    /// For persistent backends, this is the in-memory cache size.
    fn storage_size_bytes(&self) -> usize;

    /// Get the storage backend type.
    ///
    /// Returns the enum variant identifying this implementation.
    fn backend_type(&self) -> TeleologicalStorageBackend;

    // ==================== Persistence ====================

    /// Flush all pending writes to durable storage.
    ///
    /// For in-memory stores, this is a no-op.
    /// For persistent stores, ensures all data is written to disk.
    ///
    /// # Errors
    /// - `CoreError::StorageError` - Flush failure
    async fn flush(&self) -> CoreResult<()>;

    /// Create a checkpoint of the current store state.
    ///
    /// Returns the path to the checkpoint directory/file.
    /// Checkpoints enable point-in-time recovery.
    ///
    /// # Returns
    /// Path to the created checkpoint.
    ///
    /// # Errors
    /// - `CoreError::StorageError` - Checkpoint creation failure
    async fn checkpoint(&self) -> CoreResult<PathBuf>;

    /// Restore store state from a checkpoint.
    ///
    /// Replaces current state with checkpoint data.
    /// **WARNING**: Destructive operation - current data is lost.
    ///
    /// # Arguments
    /// * `checkpoint_path` - Path to the checkpoint to restore
    ///
    /// # Errors
    /// - `CoreError::StorageError` - Restore failure
    /// - `CoreError::ConfigError` - Invalid checkpoint path
    async fn restore(&self, checkpoint_path: &Path) -> CoreResult<()>;

    /// Compact the storage to reclaim space.
    ///
    /// Removes soft-deleted entries and defragments storage.
    /// For RocksDB, triggers manual compaction.
    ///
    /// # Errors
    /// - `CoreError::StorageError` - Compaction failure
    async fn compact(&self) -> CoreResult<()>;

    // ==================== Content Storage (TASK-CONTENT-003) ====================
    // See `defaults.rs` for detailed documentation on default implementations.

    /// Store content text associated with a fingerprint.
    /// Default: Returns unsupported error. Override for content-capable backends.
    async fn store_content(&self, id: Uuid, content: &str) -> CoreResult<()>;

    /// Retrieve content text for a fingerprint.
    /// Default: Returns None. Override for content-capable backends.
    async fn get_content(&self, id: Uuid) -> CoreResult<Option<String>>;

    /// Delete content for a fingerprint.
    /// Default: Returns false. Override for content-capable backends.
    async fn delete_content(&self, id: Uuid) -> CoreResult<bool>;

    /// Batch retrieve content for multiple fingerprints.
    /// Default: Returns vec of None. Override for batch-optimized retrieval.
    async fn get_content_batch(&self, ids: &[Uuid]) -> CoreResult<Vec<Option<String>>>;
}
