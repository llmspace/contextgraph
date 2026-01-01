//! Edge CRUD operations for RocksDB backend.
//!
//! Provides store, get, update, and delete operations for GraphEdges.
//! Supports efficient prefix scans for outgoing edges.

use rocksdb::IteratorMode;

use crate::column_families::cf_names;
use crate::serialization::{deserialize_edge, serialize_edge};
use context_graph_core::marblestone::EdgeType;
use context_graph_core::types::{GraphEdge, NodeId};

use super::core::RocksDbMemex;
use super::error::StorageError;
use super::helpers::{format_edge_key, format_edge_prefix};

impl RocksDbMemex {
    /// Stores a GraphEdge with composite key for efficient lookups.
    ///
    /// Key format: source_uuid_bytes (16) + target_uuid_bytes (16) + edge_type_byte (1) = 33 bytes
    /// Preserves all 13 Marblestone fields through bincode serialization.
    ///
    /// # Performance
    /// Target: <1ms p95 latency
    ///
    /// # Errors
    /// - `StorageError::Serialization` if serialization fails
    /// - `StorageError::WriteFailed` if RocksDB write fails
    /// - `StorageError::ColumnFamilyNotFound` if edges CF missing
    pub fn store_edge(&self, edge: &GraphEdge) -> Result<(), StorageError> {
        let cf_edges = self.get_cf(cf_names::EDGES)?;
        let key = format_edge_key(&edge.source_id, &edge.target_id, edge.edge_type);
        let value = serialize_edge(edge)?;

        self.db
            .put_cf(cf_edges, &key, &value)
            .map_err(|e| StorageError::WriteFailed(e.to_string()))?;

        Ok(())
    }

    /// Retrieves a GraphEdge by source, target, and edge type.
    ///
    /// # Performance
    /// Target: <500μs p95 latency
    ///
    /// # Errors
    /// - `StorageError::NotFound` if edge doesn't exist
    /// - `StorageError::Serialization` if deserialization fails
    /// - `StorageError::ReadFailed` if RocksDB read fails
    pub fn get_edge(
        &self,
        source_id: &NodeId,
        target_id: &NodeId,
        edge_type: EdgeType,
    ) -> Result<GraphEdge, StorageError> {
        let cf_edges = self.get_cf(cf_names::EDGES)?;
        let key = format_edge_key(source_id, target_id, edge_type);

        let value = self
            .db
            .get_cf(cf_edges, &key)
            .map_err(|e| StorageError::ReadFailed(e.to_string()))?
            .ok_or_else(|| StorageError::NotFound {
                id: format!("edge:{}:{}:{:?}", source_id, target_id, edge_type),
            })?;

        deserialize_edge(&value).map_err(StorageError::from)
    }

    /// Updates an existing GraphEdge.
    ///
    /// Same as store - RocksDB overwrites existing keys.
    /// DOES NOT verify edge exists first (use get_edge if verification needed).
    ///
    /// # Errors
    /// - `StorageError::Serialization` if serialization fails
    /// - `StorageError::WriteFailed` if RocksDB write fails
    pub fn update_edge(&self, edge: &GraphEdge) -> Result<(), StorageError> {
        // Same as store - RocksDB overwrites existing keys
        self.store_edge(edge)
    }

    /// Deletes a GraphEdge.
    ///
    /// Note: Does NOT return NotFound if edge doesn't exist (RocksDB delete is idempotent).
    ///
    /// # Errors
    /// - `StorageError::WriteFailed` if RocksDB delete fails
    /// - `StorageError::ColumnFamilyNotFound` if edges CF missing
    pub fn delete_edge(
        &self,
        source_id: &NodeId,
        target_id: &NodeId,
        edge_type: EdgeType,
    ) -> Result<(), StorageError> {
        let cf_edges = self.get_cf(cf_names::EDGES)?;
        let key = format_edge_key(source_id, target_id, edge_type);

        self.db
            .delete_cf(cf_edges, &key)
            .map_err(|e| StorageError::WriteFailed(e.to_string()))?;

        Ok(())
    }

    /// Gets all outgoing edges from a source node.
    ///
    /// Uses prefix scan for efficiency - O(n) where n = number of outgoing edges.
    /// The edges CF is configured with a 16-byte prefix extractor for optimal performance.
    ///
    /// # Performance
    /// Efficient prefix scan - only iterates over edges from this source.
    ///
    /// # Errors
    /// - `StorageError::Serialization` if any edge deserialization fails
    /// - `StorageError::ReadFailed` if RocksDB iteration fails
    pub fn get_edges_from(&self, source_id: &NodeId) -> Result<Vec<GraphEdge>, StorageError> {
        let cf_edges = self.get_cf(cf_names::EDGES)?;
        let prefix = format_edge_prefix(source_id);
        let mut edges = Vec::new();

        // Use prefix iterator for efficient scanning
        let iter = self.db.prefix_iterator_cf(cf_edges, &prefix);

        for item in iter {
            let (key, value) = item.map_err(|e| StorageError::ReadFailed(e.to_string()))?;

            // Check that key still starts with our prefix (RocksDB may return more)
            if !key.starts_with(&prefix) {
                break;
            }

            let edge = deserialize_edge(&value)?;
            edges.push(edge);
        }

        Ok(edges)
    }

    /// Gets all incoming edges to a target node.
    ///
    /// Uses full scan with filter - O(E) where E = total edges in database.
    /// Less efficient than get_edges_from() - no reverse index in this implementation.
    ///
    /// # Performance
    /// Full table scan - consider adding a reverse index for high-volume use cases.
    ///
    /// # Errors
    /// - `StorageError::Serialization` if any edge deserialization fails
    /// - `StorageError::ReadFailed` if RocksDB iteration fails
    pub fn get_edges_to(&self, target_id: &NodeId) -> Result<Vec<GraphEdge>, StorageError> {
        let cf_edges = self.get_cf(cf_names::EDGES)?;
        let mut edges = Vec::new();

        // Full scan with filter (no reverse index)
        let iter = self.db.iterator_cf(cf_edges, IteratorMode::Start);

        for item in iter {
            let (_key, value) = item.map_err(|e| StorageError::ReadFailed(e.to_string()))?;
            let edge = deserialize_edge(&value)?;

            if &edge.target_id == target_id {
                edges.push(edge);
            }
        }

        Ok(edges)
    }
}
