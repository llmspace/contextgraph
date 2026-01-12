//! Content and ego node storage operations for the in-memory teleological store.
//!
//! This module implements content text storage (TASK-CONTENT-004) and
//! SELF_EGO_NODE persistence (TASK-GWT-P1-001).

use tracing::{debug, error, info};
use uuid::Uuid;

use super::InMemoryTeleologicalStore;
use crate::error::{CoreError, CoreResult};
use crate::gwt::ego_node::SelfEgoNode;

impl InMemoryTeleologicalStore {
    /// Store content text for a fingerprint.
    pub async fn store_content_impl(&self, id: Uuid, content: &str) -> CoreResult<()> {
        self.content.insert(id, content.to_string());
        debug!(
            fingerprint_id = %id,
            content_size = content.len(),
            "Content stored"
        );
        Ok(())
    }

    /// Retrieve content text for a fingerprint.
    pub async fn get_content_impl(&self, id: Uuid) -> CoreResult<Option<String>> {
        Ok(self.content.get(&id).map(|r| r.clone()))
    }

    /// Retrieve content text for multiple fingerprints.
    pub async fn get_content_batch_impl(&self, ids: &[Uuid]) -> CoreResult<Vec<Option<String>>> {
        Ok(ids
            .iter()
            .map(|id| self.content.get(id).map(|r| r.clone()))
            .collect())
    }

    /// Delete content text for a fingerprint.
    pub async fn delete_content_impl(&self, id: Uuid) -> CoreResult<bool> {
        let removed = self.content.remove(&id).is_some();
        if removed {
            debug!(fingerprint_id = %id, "Content deleted");
        }
        Ok(removed)
    }

    /// Save the singleton SELF_EGO_NODE to in-memory storage.
    pub async fn save_ego_node_impl(&self, ego_node: &SelfEgoNode) -> CoreResult<()> {
        match self.ego_node.write() {
            Ok(mut guard) => {
                info!(
                    "Saving SELF_EGO_NODE id={} ({} identity snapshots)",
                    ego_node.id,
                    ego_node.identity_trajectory.len()
                );
                *guard = Some(ego_node.clone());
                Ok(())
            }
            Err(e) => {
                error!("Failed to acquire ego_node write lock: {}", e);
                Err(CoreError::Internal(format!(
                    "Failed to save ego node: lock poisoned ({})",
                    e
                )))
            }
        }
    }

    /// Load the singleton SELF_EGO_NODE from in-memory storage.
    pub async fn load_ego_node_impl(&self) -> CoreResult<Option<SelfEgoNode>> {
        match self.ego_node.read() {
            Ok(guard) => {
                if let Some(ref ego) = *guard {
                    info!(
                        "Loaded SELF_EGO_NODE id={} ({} identity snapshots)",
                        ego.id,
                        ego.identity_trajectory.len()
                    );
                }
                Ok(guard.clone())
            }
            Err(e) => {
                error!("Failed to acquire ego_node read lock: {}", e);
                Err(CoreError::Internal(format!(
                    "Failed to load ego node: lock poisoned ({})",
                    e
                )))
            }
        }
    }
}
