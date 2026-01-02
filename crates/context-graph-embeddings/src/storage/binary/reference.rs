//! Zero-copy reference to FusedEmbedding in memory-mapped buffer.

use crate::types::dimensions::{FUSED_OUTPUT, NUM_EXPERTS, TOP_K_EXPERTS};
use crate::types::{AuxiliaryEmbeddingData, FusedEmbedding};

use super::types::EmbeddingHeader;

/// Zero-copy reference to FusedEmbedding in memory-mapped buffer.
///
/// All data is borrowed from the underlying buffer - no heap allocation.
/// NOTE: Vector values are big-endian and need byte-swapping on read.
pub struct FusedEmbeddingRef<'a> {
    pub(crate) header: &'a EmbeddingHeader,
    /// Raw vector bytes (big-endian f32s), use `vector()` method for conversion
    pub(crate) vector_bytes: &'a [u8],
    /// Raw weights bytes (big-endian f32s), use `expert_weights()` method for conversion
    pub(crate) weights_bytes: &'a [u8],
    /// Selected experts bytes
    pub(crate) selected_bytes: &'a [u8],
    pub(crate) aux_data: Option<&'a [u8]>,
}

impl<'a> FusedEmbeddingRef<'a> {
    /// Get vector with byte-swapping from big-endian.
    ///
    /// This allocates a new array but does not allocate on heap
    /// (stack allocation for the fixed-size array).
    pub fn vector(&self) -> [f32; FUSED_OUTPUT] {
        let mut result = [0.0f32; FUSED_OUTPUT];
        for (i, val) in result.iter_mut().enumerate() {
            let offset = i * 4;
            *val = f32::from_be_bytes([
                self.vector_bytes[offset],
                self.vector_bytes[offset + 1],
                self.vector_bytes[offset + 2],
                self.vector_bytes[offset + 3],
            ]);
        }
        result
    }

    /// Get vector as Vec<f32> (heap allocation).
    pub fn vector_vec(&self) -> Vec<f32> {
        self.vector().to_vec()
    }

    /// Get expert weights with byte-swapping from big-endian.
    pub fn expert_weights(&self) -> [f32; NUM_EXPERTS] {
        let mut result = [0.0f32; NUM_EXPERTS];
        for (i, val) in result.iter_mut().enumerate() {
            let offset = i * 4;
            *val = f32::from_be_bytes([
                self.weights_bytes[offset],
                self.weights_bytes[offset + 1],
                self.weights_bytes[offset + 2],
                self.weights_bytes[offset + 3],
            ]);
        }
        result
    }

    /// Get selected experts.
    #[inline]
    pub fn selected_experts(&self) -> [u8; TOP_K_EXPERTS] {
        let mut result = [0u8; TOP_K_EXPERTS];
        result.copy_from_slice(&self.selected_bytes[..TOP_K_EXPERTS]);
        result
    }

    /// Convert to owned FusedEmbedding (allocates).
    pub fn to_owned(&self) -> FusedEmbedding {
        let vector = self.vector().to_vec();
        let aux_data = self
            .aux_data
            .and_then(|blob| AuxiliaryEmbeddingData::from_blob(blob).ok());

        FusedEmbedding {
            vector,
            expert_weights: self.expert_weights(),
            selected_experts: self.selected_experts(),
            pipeline_latency_us: u64::from_be(self.header.pipeline_latency_us),
            content_hash: u64::from_be(self.header.content_hash),
            aux_data,
        }
    }

    /// Get content hash.
    #[inline]
    pub fn content_hash(&self) -> u64 {
        u64::from_be(self.header.content_hash)
    }

    /// Get pipeline latency in microseconds.
    #[inline]
    pub fn pipeline_latency_us(&self) -> u64 {
        u64::from_be(self.header.pipeline_latency_us)
    }

    /// Check if embedding has auxiliary data.
    #[inline]
    pub fn has_aux_data(&self) -> bool {
        self.aux_data.is_some()
    }

    /// Get raw aux_data bytes (for deferred parsing).
    #[inline]
    pub fn aux_data_bytes(&self) -> Option<&'a [u8]> {
        self.aux_data
    }

    /// Get header reference.
    #[inline]
    pub fn header(&self) -> &EmbeddingHeader {
        self.header
    }

    /// Get raw vector bytes (big-endian).
    #[inline]
    pub fn vector_bytes_raw(&self) -> &'a [u8] {
        self.vector_bytes
    }

    /// Get raw weights bytes (big-endian).
    #[inline]
    pub fn weights_bytes_raw(&self) -> &'a [u8] {
        self.weights_bytes
    }
}
