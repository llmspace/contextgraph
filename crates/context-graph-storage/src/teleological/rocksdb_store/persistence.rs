//! Batch, statistics, and persistence operations.
//!
//! Contains batch store/retrieve, count/stats, flush/checkpoint/compact,
//! and Johari listing operations.

use std::path::PathBuf;

use tracing::{debug, error, info, warn};
use uuid::Uuid;

use context_graph_core::error::{CoreError, CoreResult};
use context_graph_core::traits::TeleologicalStorageBackend;
use context_graph_core::types::fingerprint::{JohariFingerprint, TeleologicalFingerprint};

use crate::teleological::column_families::{
    CF_FINGERPRINTS, QUANTIZED_EMBEDDER_CFS, TELEOLOGICAL_CFS,
};
use crate::teleological::schema::parse_fingerprint_key;
use crate::teleological::serialization::deserialize_teleological_fingerprint;

use super::helpers::get_aggregate_dominant_quadrant;
use super::store::RocksDbTeleologicalStore;
use super::types::TeleologicalStoreError;

// ============================================================================
// Batch Operations
// ============================================================================

impl RocksDbTeleologicalStore {
    /// Store batch of fingerprints (internal async wrapper).
    pub(crate) async fn store_batch_async(
        &self,
        fingerprints: Vec<TeleologicalFingerprint>,
    ) -> CoreResult<Vec<Uuid>> {
        debug!("Storing batch of {} fingerprints", fingerprints.len());

        let mut ids = Vec::with_capacity(fingerprints.len());

        for fp in fingerprints {
            let id = fp.id;
            self.store_fingerprint_internal(&fp)?;
            ids.push(id);
        }

        info!("Stored batch of {} fingerprints", ids.len());
        Ok(ids)
    }

    /// Retrieve batch of fingerprints (internal async wrapper).
    pub(crate) async fn retrieve_batch_async(
        &self,
        ids: &[Uuid],
    ) -> CoreResult<Vec<Option<TeleologicalFingerprint>>> {
        debug!("Retrieving batch of {} fingerprints", ids.len());

        let mut results = Vec::with_capacity(ids.len());

        for &id in ids {
            let fp = self.retrieve_async(id).await?;
            results.push(fp);
        }

        Ok(results)
    }
}

// ============================================================================
// Statistics Operations
// ============================================================================

impl RocksDbTeleologicalStore {
    /// Count fingerprints (internal async wrapper).
    pub(crate) async fn count_async(&self) -> CoreResult<usize> {
        // Check cache first
        if let Ok(cached) = self.fingerprint_count.read() {
            if let Some(count) = *cached {
                return Ok(count);
            }
        }

        // Count by iterating
        let cf = self.get_cf(CF_FINGERPRINTS)?;
        let iter = self.db.iterator_cf(cf, rocksdb::IteratorMode::Start);

        let mut count = 0;
        for item in iter {
            let (key, _) = item.map_err(|e| {
                TeleologicalStoreError::rocksdb_op("iterate", CF_FINGERPRINTS, None, e)
            })?;
            let id = parse_fingerprint_key(&key);

            if !self.is_soft_deleted(&id) {
                count += 1;
            }
        }

        // Cache the result
        if let Ok(mut cached) = self.fingerprint_count.write() {
            *cached = Some(count);
        }

        Ok(count)
    }

    /// Count fingerprints by Johari quadrant (internal async wrapper).
    pub(crate) async fn count_by_quadrant_async(&self) -> CoreResult<[usize; 4]> {
        let cf = self.get_cf(CF_FINGERPRINTS)?;
        let iter = self.db.iterator_cf(cf, rocksdb::IteratorMode::Start);

        let mut counts = [0usize; 4];

        for item in iter {
            let (key, value) = item.map_err(|e| {
                TeleologicalStoreError::rocksdb_op("iterate", CF_FINGERPRINTS, None, e)
            })?;
            let id = parse_fingerprint_key(&key);

            if self.is_soft_deleted(&id) {
                continue;
            }

            let fp = deserialize_teleological_fingerprint(&value);
            let quadrant_idx = get_aggregate_dominant_quadrant(&fp.johari);
            counts[quadrant_idx] += 1;
        }

        Ok(counts)
    }

    /// Get storage size in bytes.
    pub(crate) fn storage_size_bytes_internal(&self) -> usize {
        let mut total = 0usize;

        for cf_name in TELEOLOGICAL_CFS {
            if let Ok(cf) = self.get_cf(cf_name) {
                if let Ok(Some(size)) = self
                    .db
                    .property_int_value_cf(cf, "rocksdb.estimate-live-data-size")
                {
                    total += size as usize;
                }
            }
        }

        for cf_name in QUANTIZED_EMBEDDER_CFS {
            if let Ok(cf) = self.get_cf(cf_name) {
                if let Ok(Some(size)) = self
                    .db
                    .property_int_value_cf(cf, "rocksdb.estimate-live-data-size")
                {
                    total += size as usize;
                }
            }
        }

        total
    }

    /// Get backend type.
    pub(crate) fn backend_type_internal(&self) -> TeleologicalStorageBackend {
        TeleologicalStorageBackend::RocksDb
    }
}

// ============================================================================
// Persistence Operations
// ============================================================================

impl RocksDbTeleologicalStore {
    /// Flush all column families (internal async wrapper).
    pub(crate) async fn flush_async(&self) -> CoreResult<()> {
        debug!("Flushing all column families");

        for cf_name in TELEOLOGICAL_CFS {
            let cf = self.get_cf(cf_name)?;
            self.db
                .flush_cf(cf)
                .map_err(|e| TeleologicalStoreError::RocksDbOperation {
                    operation: "flush",
                    cf: cf_name,
                    key: None,
                    source: e,
                })?;
        }

        for cf_name in QUANTIZED_EMBEDDER_CFS {
            let cf = self.get_cf(cf_name)?;
            self.db
                .flush_cf(cf)
                .map_err(|e| TeleologicalStoreError::RocksDbOperation {
                    operation: "flush",
                    cf: cf_name,
                    key: None,
                    source: e,
                })?;
        }

        info!("Flushed all column families");
        Ok(())
    }

    /// Create checkpoint (internal async wrapper).
    pub(crate) async fn checkpoint_async(&self) -> CoreResult<PathBuf> {
        let checkpoint_path = self.path.join("checkpoints").join(format!(
            "checkpoint_{}",
            chrono::Utc::now().format("%Y%m%d_%H%M%S")
        ));

        debug!("Creating checkpoint at {:?}", checkpoint_path);

        std::fs::create_dir_all(&checkpoint_path).map_err(|e| {
            CoreError::StorageError(format!("Failed to create checkpoint directory: {}", e))
        })?;

        let checkpoint = rocksdb::checkpoint::Checkpoint::new(&self.db).map_err(|e| {
            TeleologicalStoreError::CheckpointFailed {
                message: e.to_string(),
            }
        })?;

        checkpoint
            .create_checkpoint(&checkpoint_path)
            .map_err(|e| TeleologicalStoreError::CheckpointFailed {
                message: e.to_string(),
            })?;

        info!("Created checkpoint at {:?}", checkpoint_path);
        Ok(checkpoint_path)
    }

    /// Restore from checkpoint (internal async wrapper).
    pub(crate) async fn restore_async(&self, checkpoint_path: &std::path::Path) -> CoreResult<()> {
        warn!(
            "Restore operation requested from {:?}. This is destructive!",
            checkpoint_path
        );

        if !checkpoint_path.exists() {
            return Err(TeleologicalStoreError::RestoreFailed {
                path: checkpoint_path.to_string_lossy().to_string(),
                message: "Checkpoint path does not exist".to_string(),
            }
            .into());
        }

        Err(TeleologicalStoreError::RestoreFailed {
            path: checkpoint_path.to_string_lossy().to_string(),
            message: "In-place restore not supported. Please restart the application with the checkpoint path.".to_string(),
        }.into())
    }

    /// Compact all column families (internal async wrapper).
    pub(crate) async fn compact_async(&self) -> CoreResult<()> {
        debug!("Starting compaction of all column families");

        for cf_name in TELEOLOGICAL_CFS {
            let cf = self.get_cf(cf_name)?;
            self.db.compact_range_cf(cf, None::<&[u8]>, None::<&[u8]>);
        }

        for cf_name in QUANTIZED_EMBEDDER_CFS {
            let cf = self.get_cf(cf_name)?;
            self.db.compact_range_cf(cf, None::<&[u8]>, None::<&[u8]>);
        }

        // Purge soft-deleted entries
        if let Ok(mut deleted) = self.soft_deleted.write() {
            for (id, _) in deleted.drain() {
                debug!("Purging soft-deleted entry {} from tracking", id);
            }
        }

        info!("Compaction complete");
        Ok(())
    }
}

// ============================================================================
// Johari Listing Operations
// ============================================================================

impl RocksDbTeleologicalStore {
    /// List fingerprints by Johari quadrant (internal async wrapper).
    pub(crate) async fn list_by_quadrant_async(
        &self,
        quadrant: usize,
        limit: usize,
    ) -> CoreResult<Vec<(Uuid, JohariFingerprint)>> {
        debug!("list_by_quadrant: quadrant={}, limit={}", quadrant, limit);

        if quadrant > 3 {
            error!("Invalid quadrant index: {} (must be 0-3)", quadrant);
            return Err(CoreError::ValidationError {
                field: "quadrant".to_string(),
                message: format!("Quadrant index must be 0-3, got {}", quadrant),
            });
        }

        let cf = self.get_cf(CF_FINGERPRINTS)?;
        let mut results = Vec::new();

        let iter = self.db.iterator_cf(cf, rocksdb::IteratorMode::Start);

        for item in iter {
            if results.len() >= limit {
                break;
            }

            let (key, value) = item.map_err(|e| {
                TeleologicalStoreError::rocksdb_op("iterate", CF_FINGERPRINTS, None, e)
            })?;

            let id = parse_fingerprint_key(&key);

            if self.is_soft_deleted(&id) {
                continue;
            }

            let fp = deserialize_teleological_fingerprint(&value);
            let dominant = get_aggregate_dominant_quadrant(&fp.johari);

            if dominant == quadrant {
                results.push((id, fp.johari.clone()));
            }
        }

        debug!("list_by_quadrant returned {} results", results.len());
        Ok(results)
    }

    /// List all Johari fingerprints (internal async wrapper).
    pub(crate) async fn list_all_johari_async(
        &self,
        limit: usize,
    ) -> CoreResult<Vec<(Uuid, JohariFingerprint)>> {
        debug!("list_all_johari: limit={}", limit);

        let cf = self.get_cf(CF_FINGERPRINTS)?;
        let mut results = Vec::new();

        let iter = self.db.iterator_cf(cf, rocksdb::IteratorMode::Start);

        for item in iter {
            if results.len() >= limit {
                break;
            }

            let (key, value) = item.map_err(|e| {
                TeleologicalStoreError::rocksdb_op("iterate", CF_FINGERPRINTS, None, e)
            })?;

            let id = parse_fingerprint_key(&key);

            if self.is_soft_deleted(&id) {
                continue;
            }

            let fp = deserialize_teleological_fingerprint(&value);
            results.push((id, fp.johari.clone()));
        }

        debug!("list_all_johari returned {} results", results.len());
        Ok(results)
    }
}
