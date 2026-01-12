//! TeleologicalMemoryStore trait implementation for RocksDbTeleologicalStore.
//!
//! This module contains the async trait implementation that delegates to
//! the modular implementation files:
//! - `crud.rs` - CRUD operations
//! - `search.rs` - Search operations
//! - `persistence.rs` - Batch, statistics, persistence, content, ego node

use std::path::PathBuf;

use async_trait::async_trait;
use uuid::Uuid;

use context_graph_core::error::CoreResult;
use context_graph_core::gwt::ego_node::SelfEgoNode;
use context_graph_core::traits::{
    TeleologicalMemoryStore, TeleologicalSearchOptions, TeleologicalSearchResult,
    TeleologicalStorageBackend,
};
use context_graph_core::types::fingerprint::{
    JohariFingerprint, PurposeVector, SemanticFingerprint, SparseVector, TeleologicalFingerprint,
};

use super::store::RocksDbTeleologicalStore;

// ============================================================================
// TeleologicalMemoryStore Trait Implementation
// ============================================================================

#[async_trait]
impl TeleologicalMemoryStore for RocksDbTeleologicalStore {
    // ==================== CRUD Operations ====================

    async fn store(&self, fingerprint: TeleologicalFingerprint) -> CoreResult<Uuid> {
        self.store_async(fingerprint).await
    }

    async fn retrieve(&self, id: Uuid) -> CoreResult<Option<TeleologicalFingerprint>> {
        self.retrieve_async(id).await
    }

    async fn update(&self, fingerprint: TeleologicalFingerprint) -> CoreResult<bool> {
        self.update_async(fingerprint).await
    }

    async fn delete(&self, id: Uuid, soft: bool) -> CoreResult<bool> {
        self.delete_async(id, soft).await
    }

    // ==================== Search Operations ====================

    async fn search_semantic(
        &self,
        query: &SemanticFingerprint,
        options: TeleologicalSearchOptions,
    ) -> CoreResult<Vec<TeleologicalSearchResult>> {
        self.search_semantic_async(query, options).await
    }

    async fn search_purpose(
        &self,
        query: &PurposeVector,
        options: TeleologicalSearchOptions,
    ) -> CoreResult<Vec<TeleologicalSearchResult>> {
        self.search_purpose_async(query, options).await
    }

    async fn search_text(
        &self,
        text: &str,
        options: TeleologicalSearchOptions,
    ) -> CoreResult<Vec<TeleologicalSearchResult>> {
        self.search_text_async(text, options).await
    }

    async fn search_sparse(
        &self,
        sparse_query: &SparseVector,
        top_k: usize,
    ) -> CoreResult<Vec<(Uuid, f32)>> {
        self.search_sparse_async(sparse_query, top_k).await
    }

    // ==================== Batch Operations ====================

    async fn store_batch(
        &self,
        fingerprints: Vec<TeleologicalFingerprint>,
    ) -> CoreResult<Vec<Uuid>> {
        self.store_batch_async(fingerprints).await
    }

    async fn retrieve_batch(
        &self,
        ids: &[Uuid],
    ) -> CoreResult<Vec<Option<TeleologicalFingerprint>>> {
        self.retrieve_batch_async(ids).await
    }

    // ==================== Statistics ====================

    async fn count(&self) -> CoreResult<usize> {
        self.count_async().await
    }

    async fn count_by_quadrant(&self) -> CoreResult<[usize; 4]> {
        self.count_by_quadrant_async().await
    }

    fn storage_size_bytes(&self) -> usize {
        self.storage_size_bytes_internal()
    }

    fn backend_type(&self) -> TeleologicalStorageBackend {
        self.backend_type_internal()
    }

    // ==================== Persistence ====================

    async fn flush(&self) -> CoreResult<()> {
        self.flush_async().await
    }

    async fn checkpoint(&self) -> CoreResult<PathBuf> {
        self.checkpoint_async().await
    }

    async fn restore(&self, checkpoint_path: &std::path::Path) -> CoreResult<()> {
        self.restore_async(checkpoint_path).await
    }

    async fn compact(&self) -> CoreResult<()> {
        self.compact_async().await
    }

    // ==================== Johari Listing ====================

    async fn list_by_quadrant(
        &self,
        quadrant: usize,
        limit: usize,
    ) -> CoreResult<Vec<(Uuid, JohariFingerprint)>> {
        self.list_by_quadrant_async(quadrant, limit).await
    }

    async fn list_all_johari(&self, limit: usize) -> CoreResult<Vec<(Uuid, JohariFingerprint)>> {
        self.list_all_johari_async(limit).await
    }

    // ==================== Content Storage ====================

    async fn store_content(&self, id: Uuid, content: &str) -> CoreResult<()> {
        self.store_content_async(id, content).await
    }

    async fn get_content(&self, id: Uuid) -> CoreResult<Option<String>> {
        self.get_content_async(id).await
    }

    async fn get_content_batch(&self, ids: &[Uuid]) -> CoreResult<Vec<Option<String>>> {
        self.get_content_batch_async(ids).await
    }

    async fn delete_content(&self, id: Uuid) -> CoreResult<bool> {
        self.delete_content_async(id).await
    }

    // ==================== Ego Node Storage ====================

    async fn save_ego_node(&self, ego_node: &SelfEgoNode) -> CoreResult<()> {
        self.save_ego_node_async(ego_node).await
    }

    async fn load_ego_node(&self) -> CoreResult<Option<SelfEgoNode>> {
        self.load_ego_node_async().await
    }
}
