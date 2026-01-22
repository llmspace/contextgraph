//! CRUD operations for TeleologicalMemoryStore trait.
//!
//! Contains store, retrieve, update, and delete implementations.

use rocksdb::WriteBatch;
use tracing::{debug, info};
use uuid::Uuid;

use context_graph_core::error::{CoreError, CoreResult};
use context_graph_core::types::fingerprint::TeleologicalFingerprint;

use crate::teleological::column_families::{
    CF_E12_LATE_INTERACTION, CF_E13_SPLADE_INVERTED, CF_E1_MATRYOSHKA_128, CF_FINGERPRINTS,
    CF_TOPIC_PROFILES,
};
use crate::teleological::schema::{
    content_key, e12_late_interaction_key, e1_matryoshka_128_key, fingerprint_key,
    topic_profile_key,
};
use crate::teleological::serialization::deserialize_teleological_fingerprint;

use super::store::RocksDbTeleologicalStore;
use super::types::TeleologicalStoreError;

impl RocksDbTeleologicalStore {
    /// Store a fingerprint (internal async wrapper).
    pub(crate) async fn store_async(
        &self,
        fingerprint: TeleologicalFingerprint,
    ) -> CoreResult<Uuid> {
        let id = fingerprint.id;
        debug!("Storing fingerprint {}", id);

        // Store in RocksDB (primary storage)
        self.store_fingerprint_internal(&fingerprint)?;

        // Add to per-embedder indexes for O(log n) search
        self.add_to_indexes(&fingerprint)
            .map_err(|e| CoreError::IndexError(e.to_string()))?;

        Ok(id)
    }

    /// Retrieve a fingerprint (internal async wrapper).
    pub(crate) async fn retrieve_async(
        &self,
        id: Uuid,
    ) -> CoreResult<Option<TeleologicalFingerprint>> {
        debug!("Retrieving fingerprint {}", id);

        // Check soft-deleted
        if self.is_soft_deleted(&id) {
            return Ok(None);
        }

        let raw = self.get_fingerprint_raw(id)?;

        match raw {
            Some(data) => {
                let fp = deserialize_teleological_fingerprint(&data);
                Ok(Some(fp))
            }
            None => Ok(None),
        }
    }

    /// Update a fingerprint (internal async wrapper).
    pub(crate) async fn update_async(
        &self,
        fingerprint: TeleologicalFingerprint,
    ) -> CoreResult<bool> {
        let id = fingerprint.id;
        debug!("Updating fingerprint {}", id);

        // Check if exists
        let existing = self.get_fingerprint_raw(id)?;
        if existing.is_none() {
            return Ok(false);
        }

        // If updating, we need to remove old terms from inverted indexes first
        if let Some(old_data) = existing {
            let old_fp = deserialize_teleological_fingerprint(&old_data);
            let mut batch = WriteBatch::default();

            // Remove from E13 SPLADE inverted index
            self.remove_from_splade_inverted_index(&mut batch, &id, &old_fp.semantic.e13_splade)?;

            // Remove from E6 sparse inverted index (if present)
            // Per e6upgrade.md: must remove old terms before adding new ones
            if let Some(old_e6_sparse) = &old_fp.e6_sparse {
                self.remove_from_e6_sparse_inverted_index(&mut batch, &id, old_e6_sparse)?;
            }

            self.db.write(batch).map_err(|e| {
                TeleologicalStoreError::rocksdb_op(
                    "write_batch",
                    CF_E13_SPLADE_INVERTED,
                    Some(id),
                    e,
                )
            })?;
        }

        // Remove from per-embedder indexes (will be re-added with updated vectors)
        self.remove_from_indexes(id)
            .map_err(|e| CoreError::IndexError(e.to_string()))?;

        // Store updated fingerprint in RocksDB
        self.store_fingerprint_internal(&fingerprint)?;

        // Add updated fingerprint to per-embedder indexes
        self.add_to_indexes(&fingerprint)
            .map_err(|e| CoreError::IndexError(e.to_string()))?;

        Ok(true)
    }

    /// Delete a fingerprint (internal async wrapper).
    pub(crate) async fn delete_async(&self, id: Uuid, soft: bool) -> CoreResult<bool> {
        debug!("Deleting fingerprint {} (soft={})", id, soft);

        let existing = self.get_fingerprint_raw(id)?;
        if existing.is_none() {
            return Ok(false);
        }

        if soft {
            // Soft delete: mark as deleted in memory
            if let Ok(mut deleted) = self.soft_deleted.write() {
                deleted.insert(id, true);
            }
        } else {
            // Hard delete: remove from all column families
            let old_fp = deserialize_teleological_fingerprint(&existing.unwrap());
            let key = fingerprint_key(&id);

            let mut batch = WriteBatch::default();

            // Remove from fingerprints
            let cf_fp = self.get_cf(CF_FINGERPRINTS)?;
            batch.delete_cf(cf_fp, key);

            // Remove from topic profiles
            let cf_pv = self.get_cf(CF_TOPIC_PROFILES)?;
            batch.delete_cf(cf_pv, topic_profile_key(&id));

            // Remove from e1_matryoshka_128
            let cf_mat = self.get_cf(CF_E1_MATRYOSHKA_128)?;
            batch.delete_cf(cf_mat, e1_matryoshka_128_key(&id));

            // Remove from E13 SPLADE inverted index
            self.remove_from_splade_inverted_index(&mut batch, &id, &old_fp.semantic.e13_splade)?;

            // Remove from E6 sparse inverted index (if present)
            // Per e6upgrade.md: clean up E6 terms on delete
            if let Some(e6_sparse) = &old_fp.e6_sparse {
                self.remove_from_e6_sparse_inverted_index(&mut batch, &id, e6_sparse)?;
            }

            // Remove content (TASK-CONTENT-009: cascade content deletion)
            let cf_content = self.cf_content();
            batch.delete_cf(cf_content, content_key(&id));

            // Remove E12 late interaction tokens (TASK-STORAGE-P2-001)
            let cf_e12 = self.get_cf(CF_E12_LATE_INTERACTION)?;
            batch.delete_cf(cf_e12, e12_late_interaction_key(&id));

            // Remove from soft-deleted tracking
            if let Ok(mut deleted) = self.soft_deleted.write() {
                deleted.remove(&id);
            }

            self.db.write(batch).map_err(|e| {
                TeleologicalStoreError::rocksdb_op("delete_batch", CF_FINGERPRINTS, Some(id), e)
            })?;

            // Invalidate count cache
            if let Ok(mut count) = self.fingerprint_count.write() {
                *count = None;
            }

            // Remove from per-embedder indexes
            self.remove_from_indexes(id)
                .map_err(|e| CoreError::IndexError(e.to_string()))?;
        }

        info!("Deleted fingerprint {} (soft={})", id, soft);
        Ok(true)
    }
}
