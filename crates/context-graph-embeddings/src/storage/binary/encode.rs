//! Encoding functions for FusedEmbedding to binary format.

use bytemuck::bytes_of;
use std::fs::File;
use std::io::Write;

use crate::types::dimensions::{FUSED_OUTPUT, NUM_EXPERTS, TOP_K_EXPERTS};
use crate::types::FusedEmbedding;

use super::error::EncodeError;
use super::types::{CompressionType, EmbeddingHeader, EMBEDDING_BINARY_VERSION, EMBEDDING_MAGIC};

/// Binary encoder/decoder for FusedEmbedding.
///
/// Produces GDS-compatible format:
/// - 64-byte aligned header
/// - Big-endian floats for cross-platform compatibility
/// - Zero-copy decode via memory mapping
#[derive(Debug)]
pub struct EmbeddingBinaryCodec {
    /// Include auxiliary ColBERT data in output
    pub(crate) include_aux_data: bool,
    /// Compression for aux_data (v1 only supports None)
    #[allow(dead_code)]
    pub(crate) aux_compression: CompressionType,
}

impl EmbeddingBinaryCodec {
    /// Minimum buffer size without aux_data.
    /// Header(64) + Vector(6144) + Weights(32) + Selected(4) = 6244 bytes
    pub const MIN_BUFFER_SIZE: usize = 64 + (FUSED_OUTPUT * 4) + (NUM_EXPERTS * 4) + TOP_K_EXPERTS;

    /// Create codec with default settings (no aux_data).
    #[must_use]
    pub fn new() -> Self {
        Self {
            include_aux_data: false,
            aux_compression: CompressionType::None,
        }
    }

    /// Create codec with auxiliary data support.
    #[must_use]
    pub fn with_aux_data(compression: CompressionType) -> Self {
        Self {
            include_aux_data: true,
            aux_compression: compression,
        }
    }

    /// Encode FusedEmbedding to GDS-compatible bytes.
    ///
    /// # Binary Layout
    /// | Offset | Size | Field |
    /// |--------|------|-------|
    /// | 0 | 64 | EmbeddingHeader (cache-line aligned) |
    /// | 64 | 6144 | Vector: [f32; 1536] big-endian |
    /// | 6208 | 32 | ExpertWeights: [f32; 8] big-endian |
    /// | 6240 | 4 | SelectedExperts: [u8; TOP_K_EXPERTS] |
    /// | 6244 | var | AuxData (if present) |
    ///
    /// # Errors
    /// - `EncodeError::InvalidDimension` if vector dimension != 1536
    pub fn encode(&self, embedding: &FusedEmbedding) -> Result<Vec<u8>, EncodeError> {
        // Validate dimension - FAIL FAST
        if embedding.vector.len() != FUSED_OUTPUT {
            return Err(EncodeError::InvalidDimension {
                expected: FUSED_OUTPUT,
                actual: embedding.vector.len(),
            });
        }

        // Prepare aux_data if requested
        let aux_blob = if self.include_aux_data {
            embedding.aux_data.as_ref().map(|a| a.to_blob())
        } else {
            None
        };

        let aux_offset = if aux_blob.is_some() {
            Self::MIN_BUFFER_SIZE as u64
        } else {
            0
        };
        let aux_length = aux_blob.as_ref().map(|b| b.len() as u64).unwrap_or(0);

        // Build header
        let mut flags: u16 = 0;
        if aux_blob.is_some() {
            flags |= 0x01; // bit 0: has_aux_data
        }

        let header = EmbeddingHeader {
            magic: EMBEDDING_MAGIC,
            version: EMBEDDING_BINARY_VERSION.to_be(),
            flags: flags.to_be(),
            dimension: (FUSED_OUTPUT as u32).to_be(),
            num_experts: NUM_EXPERTS as u8,
            top_k: TOP_K_EXPERTS as u8,
            _reserved: [0; 2],
            content_hash: embedding.content_hash.to_be(),
            pipeline_latency_us: embedding.pipeline_latency_us.to_be(),
            aux_data_offset: aux_offset.to_be(),
            aux_data_length: aux_length.to_be(),
            _padding: [0; 16],
        };

        // Allocate buffer
        let total_size = Self::MIN_BUFFER_SIZE + aux_length as usize;
        let mut buffer = Vec::with_capacity(total_size);

        // Header (64 bytes)
        buffer.extend_from_slice(bytes_of(&header));

        // Vector (6144 bytes) - big-endian
        for &val in &embedding.vector {
            buffer.extend_from_slice(&val.to_be_bytes());
        }

        // Expert weights (32 bytes) - big-endian
        for &weight in &embedding.expert_weights {
            buffer.extend_from_slice(&weight.to_be_bytes());
        }

        // Selected experts (4 bytes)
        buffer.extend_from_slice(&embedding.selected_experts);

        // Auxiliary data (if present)
        if let Some(aux) = aux_blob {
            buffer.extend_from_slice(&aux);
        }

        debug_assert_eq!(buffer.len(), total_size, "Buffer size mismatch");
        Ok(buffer)
    }

    /// Encode directly to pre-allocated buffer (zero-copy write).
    ///
    /// # Errors
    /// - `EncodeError::BufferTooSmall` if buffer is too small
    /// - `EncodeError::InvalidDimension` if vector dimension != 1536
    pub fn encode_to_buffer(
        &self,
        embedding: &FusedEmbedding,
        buffer: &mut [u8],
    ) -> Result<usize, EncodeError> {
        let encoded = self.encode(embedding)?;
        if buffer.len() < encoded.len() {
            return Err(EncodeError::BufferTooSmall {
                needed: encoded.len(),
                available: buffer.len(),
            });
        }
        buffer[..encoded.len()].copy_from_slice(&encoded);
        Ok(encoded.len())
    }

    /// Encode to file.
    ///
    /// # Errors
    /// - `EncodeError::Io` on file write failure
    pub fn encode_to_file(
        &self,
        embedding: &FusedEmbedding,
        file: &mut File,
    ) -> Result<u64, EncodeError> {
        let encoded = self.encode(embedding)?;
        file.write_all(&encoded)?;
        Ok(encoded.len() as u64)
    }

    /// Compute serialized size for an embedding.
    #[must_use]
    pub fn serialized_size(&self, embedding: &FusedEmbedding) -> usize {
        let aux_size = if self.include_aux_data {
            embedding
                .aux_data
                .as_ref()
                .map(|a| a.to_blob().len())
                .unwrap_or(0)
        } else {
            0
        };
        Self::MIN_BUFFER_SIZE + aux_size
    }
}

impl Default for EmbeddingBinaryCodec {
    fn default() -> Self {
        Self::new()
    }
}
