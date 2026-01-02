//! Type definitions for batch encoding.
//!
//! Contains index file header structure and format constants.

use bytemuck::{Pod, Zeroable};

/// Index file header for batch embeddings.
#[repr(C)]
#[derive(Debug, Clone, Copy, Pod, Zeroable)]
pub struct EmbeddingIndexHeader {
    /// Magic bytes: "CGEI" = Context Graph Embedding Index
    pub magic: [u8; 4],
    /// Format version
    pub version: u16,
    /// Reserved
    pub _reserved: u16,
    /// Number of entries in the index
    pub entry_count: u64,
    /// Hash of associated data file (for integrity)
    pub data_file_hash: u64,
}

/// Index file magic bytes: "CGEI"
pub const INDEX_MAGIC: [u8; 4] = [0x43, 0x47, 0x45, 0x49];

/// Index file format version.
pub const INDEX_VERSION: u16 = 1;

// Compile-time assertion: index header must be exactly 24 bytes
const _INDEX_HEADER_SIZE_CHECK: () = assert!(
    std::mem::size_of::<EmbeddingIndexHeader>() == 24,
    "EmbeddingIndexHeader must be exactly 24 bytes"
);
