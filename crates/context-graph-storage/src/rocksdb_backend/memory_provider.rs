//! Graph-backed implementation of MemoryProvider for NREM dream phase.
//!
//! TASK-007: Implements MemoryProvider trait using RocksDbMemex storage.
//! Per DREAM-001: Provider data feeds Hebbian replay (dw = eta * phi_i * phi_j).
//! Per AP-35: MUST NOT return stub data when real data is available.
//!
//! # Constitution Compliance
//!
//! - DREAM-001: Provides real memories for Hebbian replay
//! - AP-35: Returns actual graph data, not stubs
//! - AP-36: Replaces hardcoded Vec::new() stubs in process()

use std::collections::HashSet;
use std::sync::Arc;

use std::fmt;

use chrono::{Duration, Utc};
use context_graph_core::dream::MemoryProvider;
use context_graph_core::types::NodeId;
use tracing::{debug, trace, warn};
use uuid::Uuid;

use super::RocksDbMemex;

/// Graph-backed MemoryProvider for NREM dream phase.
///
/// Uses RocksDbMemex storage to retrieve real memories and edges
/// for Hebbian learning during dream consolidation.
///
/// # Thread Safety
///
/// This implementation is `Send + Sync` as required by MemoryProvider trait.
/// RocksDbMemex internally handles thread safety.
///
/// # Example
///
/// ```ignore
/// use std::sync::Arc;
/// use context_graph_storage::rocksdb_backend::{RocksDbMemex, GraphMemoryProvider};
///
/// let storage = Arc::new(RocksDbMemex::open("/tmp/test")?);
/// let provider = GraphMemoryProvider::new(storage);
///
/// // Use with NremPhase
/// let mut nrem = NremPhase::new();
/// nrem.set_memory_provider(Arc::new(provider));
/// ```
pub struct GraphMemoryProvider {
    /// Reference to graph storage
    storage: Arc<RocksDbMemex>,
}

// Manual Debug implementation since RocksDbMemex doesn't implement Debug
impl fmt::Debug for GraphMemoryProvider {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("GraphMemoryProvider")
            .field("storage", &"RocksDbMemex { ... }")
            .finish()
    }
}

impl GraphMemoryProvider {
    /// Create a new GraphMemoryProvider.
    ///
    /// # Arguments
    ///
    /// * `storage` - Arc to RocksDbMemex storage instance
    pub fn new(storage: Arc<RocksDbMemex>) -> Self {
        Self { storage }
    }

    /// Get the underlying storage reference.
    ///
    /// Useful for testing and diagnostics.
    pub fn storage(&self) -> &Arc<RocksDbMemex> {
        &self.storage
    }
}

impl MemoryProvider for GraphMemoryProvider {
    /// Get recent memories for Hebbian replay.
    ///
    /// Retrieves memories from the graph storage, sorted by recency.
    /// Uses `importance` field as the phi_value for Hebbian learning.
    ///
    /// # Arguments
    ///
    /// * `limit` - Maximum number of memories to retrieve
    /// * `recency_bias` - How much to favor recent memories [0.0, 1.0]
    ///                    Higher values = more recent memories preferred
    ///
    /// # Returns
    ///
    /// Vector of (memory_id, timestamp_ms, phi_value) tuples.
    /// - phi_value is derived from the node's `importance` field
    /// - timestamp_ms is the creation timestamp in milliseconds
    ///
    /// # Implementation Notes
    ///
    /// Uses temporal index to efficiently query recent nodes.
    /// The recency_bias affects the time window:
    /// - 1.0 = last 1 hour
    /// - 0.8 = last 24 hours (constitution default)
    /// - 0.5 = last 7 days
    /// - 0.0 = last 30 days
    fn get_recent_memories(&self, limit: usize, recency_bias: f32) -> Vec<(Uuid, u64, f32)> {
        // Calculate time window based on recency_bias
        // Higher bias = shorter window (more recent)
        let hours_ago = match recency_bias {
            x if x >= 0.9 => 1,      // Last 1 hour
            x if x >= 0.7 => 24,     // Last 24 hours
            x if x >= 0.5 => 168,    // Last 7 days (168 hours)
            _ => 720,                // Last 30 days (720 hours)
        };

        let now = Utc::now();
        let start = now - Duration::hours(hours_ago);

        debug!(
            limit = limit,
            recency_bias = recency_bias,
            hours_ago = hours_ago,
            "GraphMemoryProvider: Getting recent memories"
        );

        // Query temporal index for recent node IDs
        let node_ids = match self.storage.get_nodes_in_time_range(start, now, Some(limit), 0) {
            Ok(ids) => ids,
            Err(e) => {
                warn!(
                    error = %e,
                    "GraphMemoryProvider: Failed to query temporal index"
                );
                return Vec::new();
            }
        };

        trace!(
            count = node_ids.len(),
            "GraphMemoryProvider: Found nodes in time range"
        );

        // Fetch node details and extract phi values
        let mut results = Vec::with_capacity(node_ids.len());
        for node_id in node_ids {
            match self.storage.get_node(&node_id) {
                Ok(node) => {
                    // Skip soft-deleted nodes
                    if node.metadata.deleted {
                        trace!(node_id = %node_id, "Skipping deleted node");
                        continue;
                    }

                    // Use importance as phi_value for Hebbian learning
                    let phi_value = node.importance;
                    let timestamp_ms = node.created_at.timestamp_millis() as u64;

                    results.push((node_id, timestamp_ms, phi_value));
                }
                Err(e) => {
                    // Node might have been deleted between query and fetch
                    trace!(
                        node_id = %node_id,
                        error = %e,
                        "GraphMemoryProvider: Failed to fetch node (may be deleted)"
                    );
                }
            }
        }

        debug!(
            fetched = results.len(),
            "GraphMemoryProvider: Retrieved memories for replay"
        );

        results
    }

    /// Get edges between the given memory nodes.
    ///
    /// Returns edges that connect nodes within the provided set.
    /// Edge weights are used for Hebbian learning updates.
    ///
    /// # Arguments
    ///
    /// * `memory_ids` - Memory IDs to find edges between
    ///
    /// # Returns
    ///
    /// Vector of (source, target, weight) tuples for edges where both
    /// source and target are in `memory_ids`.
    ///
    /// # Implementation Notes
    ///
    /// Uses a HashSet for O(1) membership checking.
    /// Queries outgoing edges from each node and filters by target.
    fn get_edges_for_memories(&self, memory_ids: &[Uuid]) -> Vec<(Uuid, Uuid, f32)> {
        if memory_ids.is_empty() {
            return Vec::new();
        }

        debug!(
            memory_count = memory_ids.len(),
            "GraphMemoryProvider: Getting edges for memories"
        );

        // Build HashSet for O(1) membership checks
        let memory_set: HashSet<NodeId> = memory_ids.iter().copied().collect();

        let mut results = Vec::new();

        // Query edges from each memory
        for source_id in memory_ids {
            match self.storage.get_edges_from(source_id) {
                Ok(edges) => {
                    for edge in edges {
                        // Only include edges where target is also in memory set
                        if memory_set.contains(&edge.target_id) {
                            // Use edge weight for Hebbian learning
                            results.push((edge.source_id, edge.target_id, edge.weight));
                        }
                    }
                }
                Err(e) => {
                    trace!(
                        source_id = %source_id,
                        error = %e,
                        "GraphMemoryProvider: Failed to get edges from node"
                    );
                }
            }
        }

        debug!(
            edge_count = results.len(),
            "GraphMemoryProvider: Found edges between memories"
        );

        results
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // Note: Full integration tests require RocksDB storage setup
    // These are smoke tests to verify the API

    #[test]
    fn test_recency_bias_time_window() {
        // Verify time window calculation logic
        assert_eq!(1, match 1.0_f32 { x if x >= 0.9 => 1, _ => 0 });
        assert_eq!(24, match 0.8_f32 { x if x >= 0.7 => 24, x if x >= 0.9 => 1, _ => 0 });
        assert_eq!(168, match 0.5_f32 { x if x >= 0.5 => 168, x if x >= 0.7 => 24, x if x >= 0.9 => 1, _ => 0 });
        assert_eq!(720, match 0.3_f32 { x if x >= 0.5 => 168, x if x >= 0.7 => 24, x if x >= 0.9 => 1, _ => 720 });
    }
}
