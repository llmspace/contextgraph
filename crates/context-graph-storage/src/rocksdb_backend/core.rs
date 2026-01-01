//! Core RocksDbMemex struct and database operations.
//!
//! Provides the main database wrapper with open/close and health check functionality.

use rocksdb::{Cache, ColumnFamily, Options, DB};
use std::path::Path;

use crate::column_families::{cf_names, get_column_family_descriptors};

use super::config::RocksDbConfig;
use super::error::StorageError;

/// RocksDB-backed storage implementation.
///
/// Provides persistent storage for MemoryNodes and GraphEdges with
/// optimized column families for different access patterns.
///
/// # Thread Safety
/// RocksDB's `DB` type is internally thread-safe for concurrent reads and writes.
/// This struct can be shared across threads via `Arc<RocksDbMemex>`.
///
/// # Column Families
/// Opens all 12 column families defined in `column_families.rs`.
///
/// # Example
/// ```rust,ignore
/// use context_graph_storage::rocksdb_backend::{RocksDbMemex, RocksDbConfig};
/// use tempfile::TempDir;
///
/// let tmp = TempDir::new().unwrap();
/// let db = RocksDbMemex::open(tmp.path()).expect("open failed");
/// assert!(db.health_check().is_ok());
/// ```
pub struct RocksDbMemex {
    /// The RocksDB database instance.
    pub(crate) db: DB,
    /// Shared block cache (kept alive for DB lifetime).
    #[allow(dead_code)]
    cache: Cache,
    /// Database path for reference.
    path: String,
}

impl RocksDbMemex {
    /// Open a RocksDB database at the specified path with default configuration.
    ///
    /// Creates the database and all 12 column families if they don't exist.
    ///
    /// # Arguments
    /// * `path` - Path to the database directory
    ///
    /// # Returns
    /// * `Ok(RocksDbMemex)` - Successfully opened database
    /// * `Err(StorageError::OpenFailed)` - Database could not be opened
    pub fn open<P: AsRef<Path>>(path: P) -> Result<Self, StorageError> {
        Self::open_with_config(path, RocksDbConfig::default())
    }

    /// Open a RocksDB database with custom configuration.
    ///
    /// # Arguments
    /// * `path` - Path to the database directory
    /// * `config` - Custom configuration options
    ///
    /// # Returns
    /// * `Ok(RocksDbMemex)` - Successfully opened database
    /// * `Err(StorageError::OpenFailed)` - Database could not be opened
    pub fn open_with_config<P: AsRef<Path>>(
        path: P,
        config: RocksDbConfig,
    ) -> Result<Self, StorageError> {
        let path_str = path.as_ref().to_string_lossy().to_string();

        // Create shared block cache
        let cache = Cache::new_lru_cache(config.block_cache_size);

        // Create DB options
        let mut db_opts = Options::default();
        db_opts.create_if_missing(config.create_if_missing);
        db_opts.create_missing_column_families(true);
        db_opts.set_max_open_files(config.max_open_files);

        // WAL configuration
        if !config.enable_wal {
            db_opts.set_manual_wal_flush(true);
        }

        // Get column family descriptors with optimized options
        let cf_descriptors = get_column_family_descriptors(&cache);

        // Open database with all column families
        let db = DB::open_cf_descriptors(&db_opts, &path_str, cf_descriptors).map_err(|e| {
            StorageError::OpenFailed {
                path: path_str.clone(),
                message: e.to_string(),
            }
        })?;

        Ok(Self {
            db,
            cache,
            path: path_str,
        })
    }

    /// Get a reference to a column family by name.
    ///
    /// # Arguments
    /// * `name` - Column family name (use `cf_names::*` constants)
    ///
    /// # Returns
    /// * `Ok(&ColumnFamily)` - Reference to the column family
    /// * `Err(StorageError::ColumnFamilyNotFound)` - CF doesn't exist
    pub fn get_cf(&self, name: &str) -> Result<&ColumnFamily, StorageError> {
        self.db
            .cf_handle(name)
            .ok_or_else(|| StorageError::ColumnFamilyNotFound {
                name: name.to_string(),
            })
    }

    /// Get the database path.
    ///
    /// # Returns
    /// The path where the database is stored.
    pub fn path(&self) -> &str {
        &self.path
    }

    /// Check if the database is healthy.
    ///
    /// Verifies all 12 column families are accessible.
    ///
    /// # Returns
    /// * `Ok(())` - All CFs accessible
    /// * `Err(StorageError::ColumnFamilyNotFound)` - A CF is missing
    pub fn health_check(&self) -> Result<(), StorageError> {
        for cf_name in cf_names::ALL {
            self.get_cf(cf_name)?;
        }
        Ok(())
    }

    /// Flush all column families to disk.
    ///
    /// Forces all buffered writes to be persisted.
    ///
    /// # Returns
    /// * `Ok(())` - All CFs flushed successfully
    /// * `Err(StorageError::FlushFailed)` - Flush operation failed
    pub fn flush_all(&self) -> Result<(), StorageError> {
        for cf_name in cf_names::ALL {
            let cf = self.get_cf(cf_name)?;
            self.db
                .flush_cf(cf)
                .map_err(|e| StorageError::FlushFailed(e.to_string()))?;
        }
        Ok(())
    }

    /// Get a reference to the underlying RocksDB instance.
    ///
    /// Use this for advanced operations not covered by the high-level API.
    /// Be careful not to violate data invariants.
    pub fn db(&self) -> &DB {
        &self.db
    }
}

// DB is automatically closed when RocksDbMemex is dropped (RocksDB's Drop impl)
