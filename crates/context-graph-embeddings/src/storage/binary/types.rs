//! Binary format types and constants for FusedEmbedding.
//!
//! This module defines the GDS-compatible binary header and compression types.

use bytemuck::{Pod, Zeroable};

/// Binary format version. Increment when format changes.
/// Version 1: Initial GDS-compatible format with 64-byte header.
pub const EMBEDDING_BINARY_VERSION: u16 = 1;

/// Magic bytes for file identification: "CGEB" = Context Graph Embedding Binary
pub const EMBEDDING_MAGIC: [u8; 4] = [0x43, 0x47, 0x45, 0x42];

/// Fixed-size binary header (64 bytes, cache-line aligned).
/// MUST remain exactly 64 bytes for GDS compatibility.
///
/// Layout (all multi-byte values stored big-endian in the encoded stream):
/// - [0..4] magic: "CGEB"
/// - [4..6] version: u16
/// - [6..8] flags: u16
/// - [8..12] dimension: u32
/// - [12] num_experts: u8
/// - [13] top_k: u8
/// - [14..16] reserved: [u8; 2]
/// - [16..24] content_hash: u64
/// - [24..32] pipeline_latency_us: u64
/// - [32..40] aux_data_offset: u64
/// - [40..48] aux_data_length: u64
/// - [48..64] padding: [u8; 16]
#[repr(C)]
#[derive(Debug, Clone, Copy, Pod, Zeroable)]
pub struct EmbeddingHeader {
    /// Magic bytes: "CGEB" (0x43 0x47 0x45 0x42)
    pub magic: [u8; 4],
    /// Format version (big-endian)
    pub version: u16,
    /// Flags: bit 0 = has_aux_data, bit 1 = compressed_aux, bits 2-15 reserved
    pub flags: u16,
    /// Vector dimension (1536 for FusedEmbedding)
    pub dimension: u32,
    /// Number of experts (8)
    pub num_experts: u8,
    /// Top-K experts selected (4 per constitution.yaml)
    pub top_k: u8,
    /// Reserved for future use
    pub _reserved: [u8; 2],
    /// Content hash (xxHash64) for integrity verification
    pub content_hash: u64,
    /// Pipeline latency in microseconds
    pub pipeline_latency_us: u64,
    /// Auxiliary data offset from start of record (0 if none)
    pub aux_data_offset: u64,
    /// Auxiliary data length in bytes (0 if none)
    pub aux_data_length: u64,
    /// Padding to reach exactly 64 bytes
    pub _padding: [u8; 16],
}

// Compile-time assertion: header must be exactly 64 bytes
const _HEADER_SIZE_CHECK: () = assert!(
    std::mem::size_of::<EmbeddingHeader>() == 64,
    "EmbeddingHeader must be exactly 64 bytes"
);

/// Compression type for auxiliary data.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CompressionType {
    /// No compression (raw bytes)
    None,
    /// LZ4 fast compression (not implemented in v1)
    Lz4,
    /// Zstd compression (not implemented in v1)
    Zstd,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_header_is_exactly_64_bytes() {
        let size = std::mem::size_of::<EmbeddingHeader>();
        assert_eq!(size, 64);
    }

    #[test]
    fn test_header_alignment_is_suitable_for_pod() {
        let align = std::mem::align_of::<EmbeddingHeader>();
        assert!(align >= 8, "Alignment must be at least 8 bytes for u64 fields");
    }

    #[test]
    fn test_magic_bytes() {
        assert_eq!(&EMBEDDING_MAGIC, b"CGEB");
    }
}
