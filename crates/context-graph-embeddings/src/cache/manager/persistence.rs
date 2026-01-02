//! Disk persistence for cache entries.
//!
//! This module provides atomic persist/load operations for the cache manager
//! using bincode serialization with xxhash64 checksum verification.

use std::path::Path;

use tracing::error;

use crate::cache::types::CacheEntry;
use crate::error::{EmbeddingError, EmbeddingResult};

use super::core::CacheManager;
use super::serializable::SerializableCacheEntry;

// Re-export constants for tests
pub use super::serializable::{CACHE_MAGIC, CACHE_VERSION};

impl CacheManager {
    /// Persist cache to disk (bincode format).
    ///
    /// File format:
    /// - Header: magic bytes [0x43, 0x47, 0x45, 0x43], version u8, entry_count u64
    /// - Entries: bincode-serialized Vec<SerializableCacheEntry>
    /// - Footer: xxhash64 checksum of all preceding bytes
    ///
    /// # Returns
    /// * `Ok(())` - Cache persisted successfully
    /// * `Err(EmbeddingError::CacheError)` - On I/O or serialization failure
    /// * `Err(EmbeddingError::ConfigError)` - If disk_path not configured
    pub async fn persist(&self) -> EmbeddingResult<()> {
        let path = self.config.disk_path.as_ref().ok_or_else(|| {
            error!("CacheManager persist error: disk_path not configured");
            EmbeddingError::ConfigError {
                message: "disk_path not configured for persistence".to_string(),
            }
        })?;

        // Collect entries under read lock, converting to serializable format
        let entries_data: Vec<SerializableCacheEntry> = {
            let entries = self.entries.read().map_err(|e| {
                error!("CacheManager persist error: lock poisoned: {}", e);
                EmbeddingError::CacheError {
                    message: format!("Lock poisoned: {}", e),
                }
            })?;

            entries
                .iter()
                .map(|(k, v)| SerializableCacheEntry::from_cache_entry(*k, &v.embedding))
                .collect()
        };

        let entry_count = entries_data.len() as u64;

        // Build the data buffer
        let mut data = Vec::new();

        // Header: magic + version + entry_count
        data.extend_from_slice(&CACHE_MAGIC);
        data.push(CACHE_VERSION);
        data.extend_from_slice(&entry_count.to_le_bytes());

        // Serialize entries
        let entries_bytes = bincode::serialize(&entries_data).map_err(|e| {
            error!("CacheManager persist error: serialization failed: {}", e);
            EmbeddingError::SerializationError {
                message: format!("bincode serialization failed: {}", e),
            }
        })?;
        data.extend_from_slice(&entries_bytes);

        // Footer: xxhash64 checksum
        let checksum = xxhash_rust::xxh64::xxh64(&data, 0);
        data.extend_from_slice(&checksum.to_le_bytes());

        // Write to disk atomically via temp file
        let temp_path = path.with_extension("tmp");
        tokio::fs::write(&temp_path, &data)
            .await
            .map_err(|e: std::io::Error| {
                error!("CacheManager persist error: write failed: {}", e);
                EmbeddingError::CacheError {
                    message: format!("Failed to write cache file: {}", e),
                }
            })?;

        tokio::fs::rename(&temp_path, path)
            .await
            .map_err(|e: std::io::Error| {
                error!("CacheManager persist error: rename failed: {}", e);
                EmbeddingError::CacheError {
                    message: format!("Failed to rename temp cache file: {}", e),
                }
            })?;

        Ok(())
    }

    /// Load cache from disk, replacing current entries.
    ///
    /// # Returns
    /// * `Ok(())` - Cache loaded successfully
    /// * `Err(EmbeddingError::CacheError)` - On I/O, deserialization, or checksum failure
    /// * `Err(EmbeddingError::ConfigError)` - If disk_path not configured
    pub async fn load(&self) -> EmbeddingResult<()> {
        let path = self.config.disk_path.as_ref().ok_or_else(|| {
            error!("CacheManager load error: disk_path not configured");
            EmbeddingError::ConfigError {
                message: "disk_path not configured for persistence".to_string(),
            }
        })?;

        self.load_from_path(path).await
    }

    /// Load cache from a specific path.
    pub(crate) async fn load_from_path(&self, path: &Path) -> EmbeddingResult<()> {
        // Read file
        let data = tokio::fs::read(path).await.map_err(|e| {
            error!("CacheManager load error: read failed: {}", e);
            EmbeddingError::CacheError {
                message: format!("Failed to read cache file: {}", e),
            }
        })?;

        // Minimum size check: magic(4) + version(1) + count(8) + checksum(8) = 21
        if data.len() < 21 {
            error!("CacheManager load error: file too small");
            return Err(EmbeddingError::CacheError {
                message: "Cache file too small".to_string(),
            });
        }

        // Verify checksum (last 8 bytes)
        let checksum_offset = data.len() - 8;
        let stored_checksum = u64::from_le_bytes(
            data[checksum_offset..]
                .try_into()
                .map_err(|_| EmbeddingError::CacheError {
                    message: "Invalid checksum bytes".to_string(),
                })?,
        );

        let computed_checksum = xxhash_rust::xxh64::xxh64(&data[..checksum_offset], 0);

        if stored_checksum != computed_checksum {
            error!(
                "CacheManager load error: checksum mismatch (stored={:#x}, computed={:#x})",
                stored_checksum, computed_checksum
            );
            return Err(EmbeddingError::CacheError {
                message: format!(
                    "Checksum mismatch: stored={:#x}, computed={:#x}",
                    stored_checksum, computed_checksum
                ),
            });
        }

        // Verify magic bytes
        if data[0..4] != CACHE_MAGIC {
            error!("CacheManager load error: invalid magic bytes");
            return Err(EmbeddingError::CacheError {
                message: "Invalid cache file magic bytes".to_string(),
            });
        }

        // Verify version
        let version = data[4];
        if version != CACHE_VERSION {
            error!(
                "CacheManager load error: unsupported version {} (expected {})",
                version, CACHE_VERSION
            );
            return Err(EmbeddingError::CacheError {
                message: format!(
                    "Unsupported cache version {} (expected {})",
                    version, CACHE_VERSION
                ),
            });
        }

        // Parse entry count
        let entry_count = u64::from_le_bytes(
            data[5..13]
                .try_into()
                .map_err(|_| EmbeddingError::CacheError {
                    message: "Invalid entry count bytes".to_string(),
                })?,
        );

        // Deserialize entries using SerializableCacheEntry
        let entries_data: Vec<SerializableCacheEntry> =
            bincode::deserialize(&data[13..checksum_offset]).map_err(|e| {
                error!("CacheManager load error: deserialization failed: {}", e);
                EmbeddingError::SerializationError {
                    message: format!("bincode deserialization failed: {}", e),
                }
            })?;

        // Verify entry count
        if entries_data.len() as u64 != entry_count {
            error!(
                "CacheManager load error: entry count mismatch (header={}, actual={})",
                entry_count,
                entries_data.len()
            );
            return Err(EmbeddingError::CacheError {
                message: format!(
                    "Entry count mismatch: header={}, actual={}",
                    entry_count,
                    entries_data.len()
                ),
            });
        }

        // Replace current entries
        let mut entries = self.entries.write().map_err(|e| {
            error!("CacheManager load error: lock poisoned: {}", e);
            EmbeddingError::CacheError {
                message: format!("Lock poisoned: {}", e),
            }
        })?;

        entries.clear();
        self.metrics.reset();

        for serialized in entries_data {
            let (key, embedding) = serialized.into_key_and_embedding();
            let entry = CacheEntry::new(embedding);
            let size = entry.memory_size();
            entries.insert(key, entry);
            self.metrics.add_bytes(size);
        }

        Ok(())
    }
}
