//! Inverted index operations for SPLADE sparse vectors.
//!
//! Contains methods for updating and removing fingerprints from the
//! E13 SPLADE inverted index.

use rocksdb::WriteBatch;
use uuid::Uuid;

use context_graph_core::types::fingerprint::SparseVector;

use crate::teleological::column_families::CF_E13_SPLADE_INVERTED;
use crate::teleological::schema::e13_splade_inverted_key;
use crate::teleological::serialization::{deserialize_memory_id_list, serialize_memory_id_list};

use super::store::RocksDbTeleologicalStore;
use super::types::{TeleologicalStoreError, TeleologicalStoreResult};

impl RocksDbTeleologicalStore {
    /// Update the E13 SPLADE inverted index for a fingerprint.
    ///
    /// For each active term in the sparse vector, adds this fingerprint's ID
    /// to the posting list if not already present.
    pub(crate) fn update_splade_inverted_index(
        &self,
        batch: &mut WriteBatch,
        id: &Uuid,
        sparse: &SparseVector,
    ) -> TeleologicalStoreResult<()> {
        let cf_inverted = self.get_cf(CF_E13_SPLADE_INVERTED)?;

        // For each active term, update the posting list
        for &term_id in &sparse.indices {
            let term_key = e13_splade_inverted_key(term_id);

            // Read existing posting list
            let existing = self.db.get_cf(cf_inverted, term_key).map_err(|e| {
                TeleologicalStoreError::rocksdb_op("get", CF_E13_SPLADE_INVERTED, None, e)
            })?;

            let mut ids: Vec<Uuid> = match existing {
                Some(data) => deserialize_memory_id_list(&data),
                None => Vec::new(),
            };

            // Add this ID if not already present
            if !ids.contains(id) {
                ids.push(*id);
                let serialized = serialize_memory_id_list(&ids);
                batch.put_cf(cf_inverted, term_key, &serialized);
            }
        }

        Ok(())
    }

    /// Remove a fingerprint's terms from the inverted index.
    ///
    /// For each active term in the sparse vector, removes this fingerprint's ID
    /// from the posting list. Deletes the term key if the posting list becomes empty.
    pub(crate) fn remove_from_splade_inverted_index(
        &self,
        batch: &mut WriteBatch,
        id: &Uuid,
        sparse: &SparseVector,
    ) -> TeleologicalStoreResult<()> {
        let cf_inverted = self.get_cf(CF_E13_SPLADE_INVERTED)?;

        for &term_id in &sparse.indices {
            let term_key = e13_splade_inverted_key(term_id);

            let existing = self.db.get_cf(cf_inverted, term_key).map_err(|e| {
                TeleologicalStoreError::rocksdb_op("get", CF_E13_SPLADE_INVERTED, None, e)
            })?;

            if let Some(data) = existing {
                let mut ids: Vec<Uuid> = deserialize_memory_id_list(&data);
                ids.retain(|&i| i != *id);

                if ids.is_empty() {
                    batch.delete_cf(cf_inverted, term_key);
                } else {
                    let serialized = serialize_memory_id_list(&ids);
                    batch.put_cf(cf_inverted, term_key, &serialized);
                }
            }
        }

        Ok(())
    }
}
