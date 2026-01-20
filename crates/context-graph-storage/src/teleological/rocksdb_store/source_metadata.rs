//! Source metadata storage operations for RocksDbTeleologicalStore.
//!
//! Contains methods for storing and retrieving source metadata (provenance tracking).

use tracing::{debug, error, info};
use uuid::Uuid;

use context_graph_core::error::{CoreError, CoreResult};
use context_graph_core::types::SourceMetadata;

use crate::teleological::column_families::CF_SOURCE_METADATA;
use crate::teleological::schema::source_metadata_key;

use super::store::RocksDbTeleologicalStore;
use super::types::TeleologicalStoreError;

impl RocksDbTeleologicalStore {
    /// Store source metadata for a fingerprint (internal async wrapper).
    pub(crate) async fn store_source_metadata_async(
        &self,
        id: Uuid,
        metadata: &SourceMetadata,
    ) -> CoreResult<()> {
        // Serialize metadata using bincode
        let bytes = bincode::serialize(metadata).map_err(|e| {
            error!(
                "METADATA ERROR: Failed to serialize source metadata for fingerprint {}: {}",
                id, e
            );
            CoreError::Internal(format!(
                "Failed to serialize source metadata for {}: {}",
                id, e
            ))
        })?;

        let cf = self.cf_source_metadata();
        let key = source_metadata_key(&id);

        self.db.put_cf(cf, key, &bytes).map_err(|e| {
            error!(
                "ROCKSDB ERROR: Failed to store source metadata for fingerprint {}: {}",
                id, e
            );
            TeleologicalStoreError::rocksdb_op("put_source_metadata", CF_SOURCE_METADATA, Some(id), e)
        })?;

        info!(
            "Stored source metadata for fingerprint {} ({} bytes, type: {:?})",
            id,
            bytes.len(),
            metadata.source_type
        );
        Ok(())
    }

    /// Retrieve source metadata for a fingerprint (internal async wrapper).
    pub(crate) async fn get_source_metadata_async(
        &self,
        id: Uuid,
    ) -> CoreResult<Option<SourceMetadata>> {
        let key = source_metadata_key(&id);
        let cf = self.cf_source_metadata();

        match self.db.get_cf(cf, key) {
            Ok(Some(bytes)) => {
                let metadata: SourceMetadata = bincode::deserialize(&bytes).map_err(|e| {
                    error!(
                        "METADATA ERROR: Failed to deserialize source metadata for fingerprint {}. \
                         Error: {}. Bytes length: {}. This indicates data corruption.",
                        id,
                        e,
                        bytes.len()
                    );
                    CoreError::Internal(format!(
                        "Failed to deserialize source metadata for {}: {}. Data corruption detected.",
                        id, e
                    ))
                })?;
                debug!("Retrieved source metadata for fingerprint {}", id);
                Ok(Some(metadata))
            }
            Ok(None) => {
                debug!("No source metadata found for fingerprint {}", id);
                Ok(None)
            }
            Err(e) => {
                error!(
                    "ROCKSDB ERROR: Failed to read source metadata for fingerprint {}: {}",
                    id, e
                );
                Err(CoreError::StorageError(format!(
                    "Failed to read source metadata for {}: {}",
                    id, e
                )))
            }
        }
    }

    /// Batch retrieve source metadata (internal async wrapper).
    pub(crate) async fn get_source_metadata_batch_async(
        &self,
        ids: &[Uuid],
    ) -> CoreResult<Vec<Option<SourceMetadata>>> {
        if ids.is_empty() {
            return Ok(Vec::new());
        }

        debug!(
            "Batch retrieving source metadata for {} fingerprints",
            ids.len()
        );

        let cf = self.cf_source_metadata();

        let keys: Vec<_> = ids
            .iter()
            .map(|id| (cf, source_metadata_key(id).to_vec()))
            .collect();

        let results = self.db.multi_get_cf(keys);

        let mut metadata_vec = Vec::with_capacity(ids.len());
        for (i, result) in results.into_iter().enumerate() {
            match result {
                Ok(Some(bytes)) => {
                    let metadata: SourceMetadata = bincode::deserialize(&bytes).map_err(|e| {
                        error!(
                            "METADATA ERROR: Failed to deserialize batch source metadata for fingerprint {}. \
                             Index: {}, Error: {}",
                            ids[i], i, e
                        );
                        CoreError::Internal(format!(
                            "Failed to deserialize source metadata for {}: {}. Data corruption detected.",
                            ids[i], e
                        ))
                    })?;
                    metadata_vec.push(Some(metadata));
                }
                Ok(None) => metadata_vec.push(None),
                Err(e) => {
                    error!(
                        "ROCKSDB ERROR: Batch read failed at index {} (fingerprint {}): {}",
                        i, ids[i], e
                    );
                    return Err(CoreError::StorageError(format!(
                        "Failed to read source metadata batch at index {}: {}",
                        i, e
                    )));
                }
            }
        }

        let found_count = metadata_vec.iter().filter(|m| m.is_some()).count();
        debug!(
            "Batch source metadata retrieval complete: {} requested, {} found",
            ids.len(),
            found_count
        );
        Ok(metadata_vec)
    }

    /// Delete source metadata for a fingerprint (internal async wrapper).
    pub(crate) async fn delete_source_metadata_async(&self, id: Uuid) -> CoreResult<bool> {
        let key = source_metadata_key(&id);
        let cf = self.cf_source_metadata();

        let exists = match self.db.get_cf(cf, key) {
            Ok(Some(_)) => true,
            Ok(None) => {
                debug!("No source metadata to delete for fingerprint {}", id);
                return Ok(false);
            }
            Err(e) => {
                error!(
                    "ROCKSDB ERROR: Failed to check source metadata existence for fingerprint {}: {}",
                    id, e
                );
                return Err(CoreError::StorageError(format!(
                    "Failed to check source metadata existence for {}: {}",
                    id, e
                )));
            }
        };

        if exists {
            self.db.delete_cf(cf, key).map_err(|e| {
                error!(
                    "ROCKSDB ERROR: Failed to delete source metadata for fingerprint {}: {}",
                    id, e
                );
                CoreError::StorageError(format!(
                    "Failed to delete source metadata for {}: {}",
                    id, e
                ))
            })?;
            info!("Deleted source metadata for fingerprint {}", id);
        }

        Ok(exists)
    }
}
