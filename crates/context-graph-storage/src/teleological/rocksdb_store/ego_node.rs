//! Ego node storage operations for RocksDbTeleologicalStore.
//!
//! Contains methods for saving and loading the singleton SELF_EGO_NODE.

use tracing::{debug, error, info};

use context_graph_core::error::{CoreError, CoreResult};
use context_graph_core::gwt::ego_node::SelfEgoNode;

use crate::teleological::column_families::CF_EGO_NODE;
use crate::teleological::schema::ego_node_key;
use crate::teleological::serialization::{deserialize_ego_node, serialize_ego_node};

use super::store::RocksDbTeleologicalStore;
use super::types::TeleologicalStoreError;

impl RocksDbTeleologicalStore {
    /// Save ego node (internal async wrapper).
    ///
    /// Uses CF_EGO_NODE column family with fixed key "ego_node".
    /// Serialization uses bincode with version byte prefix.
    pub(crate) async fn save_ego_node_async(&self, ego_node: &SelfEgoNode) -> CoreResult<()> {
        debug!(
            "Saving SELF_EGO_NODE with id={}, purpose_vector={:?}",
            ego_node.id,
            &ego_node.purpose_vector[..3]
        );

        let serialized = serialize_ego_node(ego_node);
        let cf = self.cf_ego_node();
        let key = ego_node_key();

        self.db.put_cf(cf, key, &serialized).map_err(|e| {
            error!(
                "ROCKSDB ERROR: Failed to save SELF_EGO_NODE id={}: {}",
                ego_node.id, e
            );
            TeleologicalStoreError::rocksdb_op("put_ego_node", CF_EGO_NODE, Some(ego_node.id), e)
        })?;

        info!(
            "Saved SELF_EGO_NODE id={} ({} bytes, {} identity snapshots)",
            ego_node.id,
            serialized.len(),
            ego_node.identity_trajectory.len()
        );
        Ok(())
    }

    /// Load ego node (internal async wrapper).
    ///
    /// Returns None if no ego node has been saved yet (first run).
    pub(crate) async fn load_ego_node_async(&self) -> CoreResult<Option<SelfEgoNode>> {
        let cf = self.cf_ego_node();
        let key = ego_node_key();

        match self.db.get_cf(cf, key) {
            Ok(Some(data)) => {
                let ego_node = deserialize_ego_node(&data);
                info!(
                    "Loaded SELF_EGO_NODE id={} with {} identity snapshots",
                    ego_node.id,
                    ego_node.identity_trajectory.len()
                );
                Ok(Some(ego_node))
            }
            Ok(None) => {
                debug!("No SELF_EGO_NODE found - first run or not yet initialized");
                Ok(None)
            }
            Err(e) => {
                error!("ROCKSDB ERROR: Failed to load SELF_EGO_NODE: {}", e);
                Err(CoreError::StorageError(format!(
                    "Failed to load SELF_EGO_NODE: {}",
                    e
                )))
            }
        }
    }
}
