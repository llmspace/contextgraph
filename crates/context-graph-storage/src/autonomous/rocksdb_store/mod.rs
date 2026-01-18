//! RocksDB-backed storage for autonomous topic-based system.
//!
//! This module provides persistent storage for the autonomous system including:
//! - Singleton AutonomousConfig and AdaptiveThresholdState
//! - Lineage events for traceability
//! - Consolidation history records
//! - Memory curation state
//!
//! # FAIL FAST Policy
//!
//! **NO FALLBACKS. NO MOCK DATA. ERRORS ARE FATAL.**
//!
//! Every RocksDB operation that fails returns a detailed error with:
//! - The operation that failed
//! - The column family involved
//! - The key being accessed
//! - The underlying RocksDB error
//!
//! # Thread Safety
//!
//! The store is thread-safe for concurrent reads and writes via RocksDB's internal locking.
//!
//! # Module Structure
//!
//! - `config` - Store configuration options
//! - `error` - Error types and result alias
//! - `types` - Storage-local types (LineageEvent, ConsolidationRecord)
//! - `operations/` - Database operations by category

mod config;
mod error;
pub mod operations;
#[cfg(test)]
mod tests;
mod types;

use std::path::{Path, PathBuf};

use rocksdb::{Cache, ColumnFamily, Options, DB};
use serde::{Deserialize, Serialize};
use tracing::{debug, error, info};

use super::column_families::{get_autonomous_cf_descriptors, AUTONOMOUS_CFS};

// Re-export public types
pub use config::AutonomousStoreConfig;
pub use error::{AutonomousStoreError, AutonomousStoreResult, AUTONOMOUS_STORAGE_VERSION};
pub use types::{ConsolidationRecord, LineageEvent};

// ============================================================================
// Main Store Implementation
// ============================================================================

/// RocksDB-backed storage for autonomous topic-based system.
///
/// Provides persistent storage for all autonomous system state across 5 column families.
///
/// # Thread Safety
///
/// The store is thread-safe for concurrent access via RocksDB's internal locking.
///
/// # Example
///
/// ```ignore
/// use context_graph_storage::autonomous::RocksDbAutonomousStore;
/// use tempfile::TempDir;
///
/// let tmp = TempDir::new().unwrap();
/// let store = RocksDbAutonomousStore::open(tmp.path()).unwrap();
///
/// // Store config
/// let config = AutonomousConfig::default();
/// store.store_autonomous_config(&config).unwrap();
///
/// // Retrieve it
/// let retrieved = store.get_autonomous_config().unwrap();
/// assert!(retrieved.is_some());
/// ```
pub struct RocksDbAutonomousStore {
    /// The RocksDB database instance.
    db: DB,
    /// Shared block cache across column families.
    #[allow(dead_code)]
    cache: Cache,
    /// Database path.
    path: PathBuf,
}

impl RocksDbAutonomousStore {
    /// Open an autonomous store at the specified path with default configuration.
    ///
    /// Creates the database and all 7 column families if they don't exist.
    ///
    /// # Arguments
    ///
    /// * `path` - Path to the database directory
    ///
    /// # Returns
    ///
    /// * `Ok(RocksDbAutonomousStore)` - Successfully opened store
    /// * `Err(AutonomousStoreError)` - Open failed with detailed error
    ///
    /// # Errors
    ///
    /// - `AutonomousStoreError::OpenFailed` - Path invalid, permissions denied, or DB locked
    pub fn open<P: AsRef<Path>>(path: P) -> AutonomousStoreResult<Self> {
        Self::open_with_config(path, AutonomousStoreConfig::default())
    }

    /// Open an autonomous store with custom configuration.
    ///
    /// # Arguments
    ///
    /// * `path` - Path to the database directory
    /// * `config` - Custom configuration options
    ///
    /// # Returns
    ///
    /// * `Ok(RocksDbAutonomousStore)` - Successfully opened store
    /// * `Err(AutonomousStoreError)` - Open failed
    pub fn open_with_config<P: AsRef<Path>>(
        path: P,
        config: AutonomousStoreConfig,
    ) -> AutonomousStoreResult<Self> {
        let path_buf = path.as_ref().to_path_buf();
        let path_str = path_buf.to_string_lossy().to_string();

        info!(
            "Opening RocksDbAutonomousStore at '{}' with cache_size={}MB",
            path_str,
            config.block_cache_size / (1024 * 1024)
        );

        // Create shared block cache
        let cache = Cache::new_lru_cache(config.block_cache_size);

        // Create DB options
        let mut db_opts = Options::default();
        db_opts.create_if_missing(config.create_if_missing);
        db_opts.create_missing_column_families(true);
        db_opts.set_max_open_files(config.max_open_files);

        if !config.enable_wal {
            db_opts.set_manual_wal_flush(true);
        }

        // Get all 7 autonomous column family descriptors
        let cf_descriptors = get_autonomous_cf_descriptors(&cache);

        debug!(
            "Opening database with {} column families",
            cf_descriptors.len()
        );

        // Open database with all column families
        let db = DB::open_cf_descriptors(&db_opts, &path_str, cf_descriptors).map_err(|e| {
            error!("Failed to open RocksDB at '{}': {}", path_str, e);
            AutonomousStoreError::OpenFailed {
                path: path_str.clone(),
                message: e.to_string(),
            }
        })?;

        info!(
            "Successfully opened RocksDbAutonomousStore with {} column families",
            AUTONOMOUS_CFS.len()
        );

        Ok(Self {
            db,
            cache,
            path: path_buf,
        })
    }

    /// Get a column family handle by name.
    ///
    /// # Errors
    ///
    /// Returns `AutonomousStoreError::ColumnFamilyNotFound` if CF doesn't exist.
    pub(crate) fn get_cf(&self, name: &str) -> AutonomousStoreResult<&ColumnFamily> {
        self.db
            .cf_handle(name)
            .ok_or_else(|| AutonomousStoreError::ColumnFamilyNotFound {
                name: name.to_string(),
            })
    }

    /// Serialize a value with version prefix.
    pub(crate) fn serialize_with_version<T: Serialize>(
        value: &T,
    ) -> AutonomousStoreResult<Vec<u8>> {
        let mut result = vec![AUTONOMOUS_STORAGE_VERSION];
        let encoded =
            bincode::serialize(value).map_err(|e| AutonomousStoreError::Serialization {
                type_name: std::any::type_name::<T>(),
                message: e.to_string(),
            })?;
        result.extend(encoded);
        Ok(result)
    }

    /// Deserialize a value with version check.
    pub(crate) fn deserialize_with_version<T: for<'de> Deserialize<'de>>(
        data: &[u8],
        cf: &'static str,
        key: &str,
    ) -> AutonomousStoreResult<T> {
        if data.is_empty() {
            return Err(AutonomousStoreError::Deserialization {
                cf,
                key: key.to_string(),
                message: "Empty data".to_string(),
            });
        }

        let version = data[0];
        if version != AUTONOMOUS_STORAGE_VERSION {
            return Err(AutonomousStoreError::VersionMismatch {
                cf,
                expected: AUTONOMOUS_STORAGE_VERSION,
                actual: version,
            });
        }

        bincode::deserialize(&data[1..]).map_err(|e| AutonomousStoreError::Deserialization {
            cf,
            key: key.to_string(),
            message: e.to_string(),
        })
    }

    /// Get the database path.
    pub fn path(&self) -> &Path {
        &self.path
    }

    /// Health check: verify all column families are accessible.
    pub fn health_check(&self) -> AutonomousStoreResult<()> {
        for cf_name in AUTONOMOUS_CFS {
            self.get_cf(cf_name)?;
        }
        Ok(())
    }
}
