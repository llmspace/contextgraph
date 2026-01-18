//! Node CRUD operations for RocksDB backend.
//!
//! Provides store, get, update, and delete operations for MemoryNodes.
//! All operations maintain index consistency across column families.

use std::collections::HashSet;

use rocksdb::WriteBatch;

use crate::column_families::cf_names;
use crate::serialization::{deserialize_node, serialize_embedding, serialize_node, serialize_uuid};
use context_graph_core::types::MemoryNode;

use super::core::RocksDbMemex;
use super::error::StorageError;
use super::helpers::{format_source_key, format_tag_key, format_temporal_key};

impl RocksDbMemex {
    /// Stores a MemoryNode atomically across all relevant column families.
    ///
    /// Writes to: nodes, embeddings, temporal, tags (per tag), sources
    /// Uses WriteBatch for atomicity.
    ///
    /// # Validation
    /// Calls node.validate() before storage - returns error if validation fails.
    /// FAIL FAST: Invalid nodes are never stored.
    ///
    /// # Performance
    /// Target: <1ms p95 latency
    ///
    /// # Errors
    /// - `StorageError::ValidationFailed` if node.validate() fails
    /// - `StorageError::Serialization` if serialization fails
    /// - `StorageError::WriteFailed` if RocksDB write fails
    /// - `StorageError::ColumnFamilyNotFound` if CF missing (should never happen)
    pub fn store_node(&self, node: &MemoryNode) -> Result<(), StorageError> {
        // 1. Validate node FIRST - fail fast
        node.validate()?;

        // 2. Create WriteBatch for atomic operation
        let mut batch = WriteBatch::default();

        // 3. Get all required column families
        let cf_nodes = self.get_cf(cf_names::NODES)?;
        let cf_embeddings = self.get_cf(cf_names::EMBEDDINGS)?;
        let cf_temporal = self.get_cf(cf_names::TEMPORAL)?;
        let cf_tags = self.get_cf(cf_names::TAGS)?;
        let cf_sources = self.get_cf(cf_names::SOURCES)?;

        // 4. Serialize and write to nodes CF
        let node_key = serialize_uuid(&node.id);
        let node_value = serialize_node(node)?;
        batch.put_cf(cf_nodes, node_key.as_slice(), &node_value);

        // 5. Write embedding to embeddings CF
        let embedding_value = serialize_embedding(&node.embedding);
        batch.put_cf(cf_embeddings, node_key.as_slice(), &embedding_value);

        // 6. Add to temporal index: timestamp_millis:uuid format
        let temporal_key = format_temporal_key(node.created_at, &node.id);
        batch.put_cf(cf_temporal, temporal_key.as_slice(), []);

        // 7. Add to tag indexes (one entry per tag)
        for tag in &node.metadata.tags {
            let tag_key = format_tag_key(tag, &node.id);
            batch.put_cf(cf_tags, tag_key.as_slice(), []);
        }

        // 8. Add to sources index if source present
        if let Some(source) = &node.metadata.source {
            let source_key = format_source_key(source, &node.id);
            batch.put_cf(cf_sources, source_key.as_slice(), []);
        }

        // 9. Execute atomic batch write
        self.db
            .write(batch)
            .map_err(|e| StorageError::WriteFailed(e.to_string()))?;

        Ok(())
    }

    /// Retrieves a MemoryNode by its ID.
    ///
    /// # Performance
    /// Target: <500μs p95 latency
    ///
    /// # Errors
    /// - `StorageError::NotFound` if node doesn't exist
    /// - `StorageError::Serialization` if deserialization fails
    /// - `StorageError::ReadFailed` if RocksDB read fails
    pub fn get_node(
        &self,
        id: &context_graph_core::types::NodeId,
    ) -> Result<MemoryNode, StorageError> {
        let cf_nodes = self.get_cf(cf_names::NODES)?;
        let node_key = serialize_uuid(id);

        let node_bytes = self
            .db
            .get_cf(cf_nodes, node_key.as_slice())
            .map_err(|e| StorageError::ReadFailed(e.to_string()))?
            .ok_or_else(|| StorageError::NotFound { id: id.to_string() })?;

        deserialize_node(&node_bytes).map_err(StorageError::from)
    }

    /// Updates an existing MemoryNode, maintaining index consistency.
    ///
    /// Handles index updates when tags change.
    /// DOES NOT create if node doesn't exist - returns NotFound error.
    /// FAIL FAST: Non-existent nodes cause immediate error.
    ///
    /// # Errors
    /// - `StorageError::NotFound` if node doesn't exist
    /// - `StorageError::ValidationFailed` if node.validate() fails
    /// - `StorageError::Serialization` if serialization fails
    /// - `StorageError::WriteFailed` if RocksDB write fails
    pub fn update_node(&self, node: &MemoryNode) -> Result<(), StorageError> {
        // 1. Validate node FIRST - fail fast
        node.validate()?;

        // 2. Get existing node (MUST exist, fail if not)
        let old_node = self.get_node(&node.id)?;

        // 3. Create WriteBatch for atomic operation
        let mut batch = WriteBatch::default();

        // 4. Get CFs
        let cf_nodes = self.get_cf(cf_names::NODES)?;
        let cf_embeddings = self.get_cf(cf_names::EMBEDDINGS)?;
        let cf_tags = self.get_cf(cf_names::TAGS)?;
        let cf_sources = self.get_cf(cf_names::SOURCES)?;

        let node_key = serialize_uuid(&node.id);

        // 5. Update node data
        batch.put_cf(cf_nodes, node_key.as_slice(), serialize_node(node)?);
        batch.put_cf(
            cf_embeddings,
            node_key.as_slice(),
            serialize_embedding(&node.embedding),
        );

        // 6. Handle tag changes
        let old_tags: HashSet<_> = old_node.metadata.tags.into_iter().collect();
        let new_tags: HashSet<_> = node.metadata.tags.iter().cloned().collect();

        for removed_tag in old_tags.difference(&new_tags) {
            batch.delete_cf(cf_tags, format_tag_key(removed_tag, &node.id));
        }
        for added_tag in new_tags.difference(&old_tags) {
            batch.put_cf(cf_tags, format_tag_key(added_tag, &node.id), []);
        }

        // 7. Handle source changes
        let old_source = old_node.metadata.source.as_ref();
        let new_source = node.metadata.source.as_ref();

        if old_source != new_source {
            if let Some(old_src) = old_source {
                batch.delete_cf(cf_sources, format_source_key(old_src, &node.id));
            }
            if let Some(new_src) = new_source {
                batch.put_cf(cf_sources, format_source_key(new_src, &node.id), []);
            }
        }

        // 8. Write atomically
        self.db
            .write(batch)
            .map_err(|e| StorageError::WriteFailed(e.to_string()))?;

        Ok(())
    }

    /// Deletes a MemoryNode and removes it from all indexes.
    ///
    /// # Arguments
    /// * `id` - Node ID to delete
    /// * `soft_delete` - If true, marks as deleted (SEC-06); if false, permanently removes
    ///
    /// # SEC-06 Compliance
    /// Soft delete preserves data for 30-day recovery per constitution.yaml.
    /// Soft-deleted nodes remain in the nodes CF but with `metadata.deleted = true`.
    ///
    /// # Errors
    /// - `StorageError::NotFound` if node doesn't exist
    /// - `StorageError::WriteFailed` if RocksDB write fails
    pub fn delete_node(
        &self,
        id: &context_graph_core::types::NodeId,
        soft_delete: bool,
    ) -> Result<(), StorageError> {
        // 1. Get existing node (MUST exist) - fail fast for non-existent
        let node = self.get_node(id)?;

        let mut batch = WriteBatch::default();
        let node_key = serialize_uuid(id);

        if soft_delete {
            // SEC-06: Mark as deleted, preserve data for 30-day recovery
            let mut updated_node = node.clone();
            updated_node.metadata.mark_deleted();

            let cf_nodes = self.get_cf(cf_names::NODES)?;
            batch.put_cf(
                cf_nodes,
                node_key.as_slice(),
                serialize_node(&updated_node)?,
            );
        } else {
            // Hard delete: Remove from ALL column families
            let cf_nodes = self.get_cf(cf_names::NODES)?;
            let cf_embeddings = self.get_cf(cf_names::EMBEDDINGS)?;
            let cf_temporal = self.get_cf(cf_names::TEMPORAL)?;
            let cf_tags = self.get_cf(cf_names::TAGS)?;
            let cf_sources = self.get_cf(cf_names::SOURCES)?;

            batch.delete_cf(cf_nodes, node_key.as_slice());
            batch.delete_cf(cf_embeddings, node_key.as_slice());
            batch.delete_cf(cf_temporal, format_temporal_key(node.created_at, id));

            for tag in &node.metadata.tags {
                batch.delete_cf(cf_tags, format_tag_key(tag, id));
            }

            if let Some(source) = &node.metadata.source {
                batch.delete_cf(cf_sources, format_source_key(source, id));
            }
        }

        self.db
            .write(batch)
            .map_err(|e| StorageError::WriteFailed(e.to_string()))?;

        Ok(())
    }
}
