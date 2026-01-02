//! Batch encoder for efficient multi-embedding serialization.
//!
//! Provides the `BatchBinaryEncoder` for accumulating embeddings and writing
//! GDS-compatible files with 4KB page alignment.

use super::types::{EmbeddingIndexHeader, INDEX_MAGIC, INDEX_VERSION};
use crate::storage::binary::{CompressionType, EmbeddingBinaryCodec, EncodeError};
use crate::types::FusedEmbedding;
use bytemuck::bytes_of;
use std::fs::OpenOptions;
use std::io::{self, Write};
use std::path::Path;

/// Batch encoder for efficient multi-embedding serialization.
///
/// Accumulates embeddings in memory and writes GDS-compatible files
/// with 4KB page alignment for optimal I/O performance.
pub struct BatchBinaryEncoder {
    codec: EmbeddingBinaryCodec,
    buffer: Vec<u8>,
    offsets: Vec<u64>,
}

impl BatchBinaryEncoder {
    /// GDS page alignment (4KB).
    pub const PAGE_SIZE: usize = 4096;

    /// Create batch encoder with estimated capacity.
    ///
    /// # Arguments
    /// * `count` - Expected number of embeddings (for buffer pre-allocation)
    #[must_use]
    pub fn with_capacity(count: usize) -> Self {
        Self {
            codec: EmbeddingBinaryCodec::new(),
            buffer: Vec::with_capacity(count * EmbeddingBinaryCodec::MIN_BUFFER_SIZE),
            offsets: Vec::with_capacity(count),
        }
    }

    /// Create batch encoder with auxiliary data support.
    ///
    /// # Arguments
    /// * `count` - Expected number of embeddings
    #[must_use]
    pub fn with_aux_data(count: usize) -> Self {
        Self {
            codec: EmbeddingBinaryCodec::with_aux_data(CompressionType::None),
            buffer: Vec::with_capacity(count * (EmbeddingBinaryCodec::MIN_BUFFER_SIZE + 1024)),
            offsets: Vec::with_capacity(count),
        }
    }

    /// Add embedding to batch.
    ///
    /// # Errors
    /// - `EncodeError` if encoding fails (e.g., invalid dimension)
    pub fn push(&mut self, embedding: &FusedEmbedding) -> Result<(), EncodeError> {
        let offset = self.buffer.len() as u64;
        let encoded = self.codec.encode(embedding)?;
        self.offsets.push(offset);
        self.buffer.extend_from_slice(&encoded);
        Ok(())
    }

    /// Get number of embeddings in batch.
    #[inline]
    pub fn len(&self) -> usize {
        self.offsets.len()
    }

    /// Check if batch is empty.
    #[inline]
    pub fn is_empty(&self) -> bool {
        self.offsets.is_empty()
    }

    /// Get current buffer size in bytes.
    #[inline]
    pub fn buffer_size(&self) -> usize {
        self.buffer.len()
    }

    /// Finalize and get bytes with offset table.
    ///
    /// Returns (data_bytes, offsets) tuple. Note that offsets are NOT page-aligned;
    /// use `write_gds_file` for page-aligned output.
    pub fn finalize(self) -> (Vec<u8>, Vec<u64>) {
        (self.buffer, self.offsets)
    }

    /// Write GDS-compatible files: .cgeb (data) and .cgei (index).
    ///
    /// Data file is 4KB-page aligned for optimal GDS performance.
    ///
    /// # File Layout
    ///
    /// ## Data File (.cgeb)
    /// Each embedding starts at a 4KB page boundary:
    /// ```text
    /// [0x0000] Embedding 0 (6244 bytes) + padding to 4KB
    /// [0x1000] Embedding 1 (6244 bytes) + padding to 4KB
    /// [0x2000] ...
    /// ```
    ///
    /// ## Index File (.cgei)
    /// ```text
    /// [0x00] EmbeddingIndexHeader (24 bytes)
    /// [0x18] Offset table: entry_count x u64 (big-endian)
    /// ```
    ///
    /// # Errors
    /// - `io::Error` on file write failure
    pub fn write_gds_file(&self, path: &Path) -> io::Result<()> {
        if self.is_empty() {
            return Err(io::Error::new(
                io::ErrorKind::InvalidInput,
                "Cannot write empty batch",
            ));
        }

        // Data file
        let data_path = path.with_extension("cgeb");
        let mut data_file = OpenOptions::new()
            .create(true)
            .write(true)
            .truncate(true)
            .open(&data_path)?;

        // Recompute offsets with 4KB page alignment
        let mut aligned_offsets = Vec::with_capacity(self.offsets.len());
        let mut current_offset = 0u64;

        // Write each embedding with page alignment
        for (i, original_offset) in self.offsets.iter().enumerate() {
            // Calculate embedding size
            let end_offset = if i + 1 < self.offsets.len() {
                self.offsets[i + 1]
            } else {
                self.buffer.len() as u64
            };
            let embedding_size = (end_offset - original_offset) as usize;

            // Pad to page boundary (except for first entry)
            if i > 0 {
                let padding = (Self::PAGE_SIZE - (current_offset as usize % Self::PAGE_SIZE))
                    % Self::PAGE_SIZE;
                if padding > 0 {
                    data_file.write_all(&vec![0u8; padding])?;
                    current_offset += padding as u64;
                }
            }

            aligned_offsets.push(current_offset);

            // Write embedding data
            let start = *original_offset as usize;
            let end = start + embedding_size;
            data_file.write_all(&self.buffer[start..end])?;
            current_offset += embedding_size as u64;
        }

        // Compute data file hash
        let data_hash = xxhash_rust::xxh64::xxh64(&self.buffer, 0);

        // Write index file
        self.write_index_file(path, &aligned_offsets, data_hash)?;

        Ok(())
    }

    /// Write unaligned files (faster but not GDS-optimized).
    ///
    /// Useful for testing or systems without GDS requirements.
    ///
    /// # Errors
    /// - `io::Error` on file write failure
    pub fn write_unaligned(&self, path: &Path) -> io::Result<()> {
        if self.is_empty() {
            return Err(io::Error::new(
                io::ErrorKind::InvalidInput,
                "Cannot write empty batch",
            ));
        }

        // Data file (no padding)
        let data_path = path.with_extension("cgeb");
        let mut data_file = OpenOptions::new()
            .create(true)
            .write(true)
            .truncate(true)
            .open(&data_path)?;

        data_file.write_all(&self.buffer)?;

        // Compute data file hash
        let data_hash = xxhash_rust::xxh64::xxh64(&self.buffer, 0);

        // Write index file
        self.write_index_file(path, &self.offsets, data_hash)?;

        Ok(())
    }

    /// Write index file with given offsets and data hash.
    fn write_index_file(&self, path: &Path, offsets: &[u64], data_hash: u64) -> io::Result<()> {
        let index_path = path.with_extension("cgei");
        let mut index_file = OpenOptions::new()
            .create(true)
            .write(true)
            .truncate(true)
            .open(&index_path)?;

        let index_header = EmbeddingIndexHeader {
            magic: INDEX_MAGIC,
            version: INDEX_VERSION.to_be(),
            _reserved: 0,
            entry_count: (offsets.len() as u64).to_be(),
            data_file_hash: data_hash.to_be(),
        };

        index_file.write_all(bytes_of(&index_header))?;

        // Write offset table (big-endian)
        for &offset in offsets {
            index_file.write_all(&offset.to_be_bytes())?;
        }

        Ok(())
    }
}
