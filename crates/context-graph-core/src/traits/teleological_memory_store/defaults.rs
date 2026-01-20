//! Default implementations for optional TeleologicalMemoryStore methods.
//!
//! This module provides default implementations for content storage and
//! source metadata methods. These are separated from the core trait to keep
//! the main trait file under the 500-line limit while maintaining clear organization.
//!
//! # Content Storage (TASK-CONTENT-003)
//!
//! Content storage allows associating original text content with fingerprints.
//! The default implementations return errors or empty results for backends
//! that don't support content storage.
//!
//! # Source Metadata Storage
//!
//! Source metadata allows tracking memory provenance (e.g., file path for
//! MDFileChunk memories). This enables context injection to display where
//! memories originated from.

use async_trait::async_trait;
use uuid::Uuid;

use crate::error::{CoreError, CoreResult};
use crate::types::SourceMetadata;

use super::backend::TeleologicalStorageBackend;

/// Extension trait providing default implementations for optional storage features.
///
/// This trait is automatically implemented for all types implementing
/// `TeleologicalMemoryStore`. It provides default behavior for:
/// - Content storage (store, get, delete, batch get)
///
/// Backends that support these features should override the corresponding
/// methods in `TeleologicalMemoryStore` directly.
#[allow(dead_code)] // Reserved for future extension trait implementations
#[async_trait]
pub trait TeleologicalMemoryStoreDefaults: Send + Sync {
    /// Get the storage backend type for error messages.
    fn backend_type(&self) -> TeleologicalStorageBackend;

    // ==================== Content Storage Defaults ====================

    /// Default: Store content - returns unsupported error.
    ///
    /// Content is stored separately from the fingerprint for efficiency.
    /// This allows large text content to be optionally retrieved without
    /// loading it for every search result.
    ///
    /// # Arguments
    /// * `id` - Fingerprint UUID
    /// * `content` - Original text content (max 1MB)
    ///
    /// # Errors
    /// - `CoreError::Internal` - Content storage not supported by backend
    async fn store_content_default(&self, id: Uuid, content: &str) -> CoreResult<()> {
        let _ = (id, content); // Suppress unused warnings
        Err(CoreError::Internal(format!(
            "Content storage not supported by {} backend",
            self.backend_type()
        )))
    }

    /// Default: Get content - returns None (graceful degradation).
    ///
    /// Returns the original text content that was stored with the fingerprint.
    /// Returns None if content was never stored or backend doesn't support it.
    ///
    /// # Arguments
    /// * `id` - Fingerprint UUID
    ///
    /// # Returns
    /// `None` - Backend does not support content storage.
    async fn get_content_default(&self, id: Uuid) -> CoreResult<Option<String>> {
        let _ = id; // Suppress unused warnings
        Ok(None)
    }

    /// Default: Delete content - returns false (nothing to delete).
    ///
    /// Called automatically when fingerprint is deleted (cascade delete).
    /// Can also be called directly to remove content while keeping the fingerprint.
    ///
    /// # Arguments
    /// * `id` - Fingerprint UUID
    ///
    /// # Returns
    /// `false` - No content existed to delete.
    async fn delete_content_default(&self, id: Uuid) -> CoreResult<bool> {
        let _ = id; // Suppress unused warnings
        Ok(false)
    }

    /// Default: Batch get content - calls get_content sequentially.
    ///
    /// More efficient than individual `get_content` calls for bulk retrieval.
    /// Returns Vec with Some for found content, None for not found.
    ///
    /// # Arguments
    /// * `ids` - Slice of fingerprint UUIDs
    ///
    /// # Returns
    /// Vector of `None` values (backend does not support content storage).
    async fn get_content_batch_default(&self, ids: &[Uuid]) -> CoreResult<Vec<Option<String>>> {
        // Default returns None for all IDs since content storage is not supported
        Ok(vec![None; ids.len()])
    }

    // ==================== Source Metadata Defaults ====================

    /// Default: Store source metadata - returns unsupported error.
    ///
    /// Source metadata tracks memory provenance (e.g., file path for MDFileChunk).
    ///
    /// # Arguments
    /// * `id` - Fingerprint UUID
    /// * `metadata` - Source metadata to store
    ///
    /// # Errors
    /// - `CoreError::Internal` - Source metadata storage not supported by backend
    async fn store_source_metadata_default(
        &self,
        id: Uuid,
        metadata: &SourceMetadata,
    ) -> CoreResult<()> {
        let _ = (id, metadata); // Suppress unused warnings
        Err(CoreError::Internal(format!(
            "Source metadata storage not supported by {} backend",
            self.backend_type()
        )))
    }

    /// Default: Get source metadata - returns None (graceful degradation).
    ///
    /// Returns the source metadata that was stored with the fingerprint.
    /// Returns None if metadata was never stored or backend doesn't support it.
    ///
    /// # Arguments
    /// * `id` - Fingerprint UUID
    ///
    /// # Returns
    /// `None` - Backend does not support source metadata storage.
    async fn get_source_metadata_default(&self, id: Uuid) -> CoreResult<Option<SourceMetadata>> {
        let _ = id; // Suppress unused warnings
        Ok(None)
    }

    /// Default: Delete source metadata - returns false (nothing to delete).
    ///
    /// Called automatically when fingerprint is deleted (cascade delete).
    ///
    /// # Arguments
    /// * `id` - Fingerprint UUID
    ///
    /// # Returns
    /// `false` - No metadata existed to delete.
    async fn delete_source_metadata_default(&self, id: Uuid) -> CoreResult<bool> {
        let _ = id; // Suppress unused warnings
        Ok(false)
    }

    /// Default: Batch get source metadata - returns vec of None.
    ///
    /// # Arguments
    /// * `ids` - Slice of fingerprint UUIDs
    ///
    /// # Returns
    /// Vector of `None` values (backend does not support source metadata storage).
    async fn get_source_metadata_batch_default(
        &self,
        ids: &[Uuid],
    ) -> CoreResult<Vec<Option<SourceMetadata>>> {
        // Default returns None for all IDs since source metadata storage is not supported
        Ok(vec![None; ids.len()])
    }
}
