//! Decoding functions for binary format to FusedEmbedding.

use bytemuck::try_from_bytes;

use crate::types::dimensions::{FUSED_OUTPUT, NUM_EXPERTS, TOP_K_EXPERTS};
use crate::types::{AuxiliaryEmbeddingData, FusedEmbedding};

use super::encode::EmbeddingBinaryCodec;
use super::error::DecodeError;
use super::reference::FusedEmbeddingRef;
use super::types::{EmbeddingHeader, EMBEDDING_BINARY_VERSION, EMBEDDING_MAGIC};

impl EmbeddingBinaryCodec {
    /// Decode FusedEmbedding from bytes.
    ///
    /// # Errors
    /// - `DecodeError::BufferTooShort` if bytes < MIN_BUFFER_SIZE
    /// - `DecodeError::InvalidMagic` if magic bytes don't match
    /// - `DecodeError::UnsupportedVersion` if version > current
    /// - `DecodeError::HashMismatch` if verify_hash=true and hash doesn't match
    pub fn decode(&self, bytes: &[u8]) -> Result<FusedEmbedding, DecodeError> {
        // Validate minimum size - FAIL FAST
        if bytes.len() < Self::MIN_BUFFER_SIZE {
            return Err(DecodeError::BufferTooShort {
                needed: Self::MIN_BUFFER_SIZE,
                available: bytes.len(),
            });
        }

        // Parse header
        let header = self.decode_header(bytes)?;

        // Parse vector (big-endian)
        let mut vector = Vec::with_capacity(FUSED_OUTPUT);
        for i in 0..FUSED_OUTPUT {
            let offset = 64 + i * 4;
            let val = f32::from_be_bytes([
                bytes[offset],
                bytes[offset + 1],
                bytes[offset + 2],
                bytes[offset + 3],
            ]);
            vector.push(val);
        }

        // Parse expert weights (big-endian)
        let mut expert_weights = [0.0f32; NUM_EXPERTS];
        for (i, weight) in expert_weights.iter_mut().enumerate() {
            let offset = 64 + (FUSED_OUTPUT * 4) + i * 4;
            *weight = f32::from_be_bytes([
                bytes[offset],
                bytes[offset + 1],
                bytes[offset + 2],
                bytes[offset + 3],
            ]);
        }

        // Parse selected experts
        let selected_offset = 64 + (FUSED_OUTPUT * 4) + (NUM_EXPERTS * 4);
        let mut selected_experts = [0u8; TOP_K_EXPERTS];
        for (i, expert) in selected_experts.iter_mut().enumerate() {
            *expert = bytes[selected_offset + i];
        }

        // Parse aux_data if present
        let aux_data_offset = u64::from_be(header.aux_data_offset) as usize;
        let aux_data_length = u64::from_be(header.aux_data_length) as usize;

        let aux_data = if aux_data_offset > 0 && aux_data_length > 0 {
            let end = aux_data_offset + aux_data_length;
            if bytes.len() < end {
                return Err(DecodeError::BufferTooShort {
                    needed: end,
                    available: bytes.len(),
                });
            }
            Some(
                AuxiliaryEmbeddingData::from_blob(&bytes[aux_data_offset..end])
                    .map_err(|e| DecodeError::AuxDataCorrupted(e.to_string()))?,
            )
        } else {
            None
        };

        Ok(FusedEmbedding {
            vector,
            expert_weights,
            selected_experts,
            pipeline_latency_us: u64::from_be(header.pipeline_latency_us),
            content_hash: u64::from_be(header.content_hash),
            aux_data,
        })
    }

    /// Decode header only (for seeking/filtering).
    ///
    /// # Errors
    /// - `DecodeError::BufferTooShort` if bytes < 64
    /// - `DecodeError::InvalidMagic` if magic bytes don't match
    /// - `DecodeError::UnsupportedVersion` if version > current
    pub fn decode_header(&self, bytes: &[u8]) -> Result<EmbeddingHeader, DecodeError> {
        if bytes.len() < 64 {
            return Err(DecodeError::BufferTooShort {
                needed: 64,
                available: bytes.len(),
            });
        }

        let header: &EmbeddingHeader =
            try_from_bytes(&bytes[0..64]).map_err(|_| DecodeError::InvalidMagic)?;

        // Validate magic - FAIL FAST
        if header.magic != EMBEDDING_MAGIC {
            return Err(DecodeError::InvalidMagic);
        }

        // Validate version - FAIL FAST
        let version = u16::from_be(header.version);
        if version > EMBEDDING_BINARY_VERSION {
            return Err(DecodeError::UnsupportedVersion(version));
        }

        Ok(*header)
    }

    /// Decode with zero-copy reference to memory-mapped buffer.
    ///
    /// Returns a borrowed view into the buffer - no heap allocation for the
    /// core data. Note that accessing vector/weights requires byte-swapping
    /// from big-endian.
    ///
    /// # Errors
    /// - Same as `decode_header`
    /// - `DecodeError::AlignmentError` if buffer not 64-byte aligned
    pub fn decode_zero_copy<'a>(
        &self,
        bytes: &'a [u8],
    ) -> Result<FusedEmbeddingRef<'a>, DecodeError> {
        // Validate alignment for zero-copy
        if !(bytes.as_ptr() as usize).is_multiple_of(64) {
            return Err(DecodeError::AlignmentError {
                expected: 64,
                actual: bytes.as_ptr() as usize % 64,
            });
        }

        self.decode_header(bytes)?;

        // Validate buffer size
        if bytes.len() < Self::MIN_BUFFER_SIZE {
            return Err(DecodeError::BufferTooShort {
                needed: Self::MIN_BUFFER_SIZE,
                available: bytes.len(),
            });
        }

        // Create references into buffer (zero-copy)
        let header_ref: &EmbeddingHeader =
            try_from_bytes(&bytes[0..64]).map_err(|_| DecodeError::InvalidMagic)?;

        // Vector bytes: 64..6208 (borrowed slice, conversion happens on access)
        let vector_bytes = &bytes[64..64 + FUSED_OUTPUT * 4];

        // Expert weights bytes: 6208..6240
        let weights_bytes =
            &bytes[64 + FUSED_OUTPUT * 4..64 + FUSED_OUTPUT * 4 + NUM_EXPERTS * 4];

        // Selected experts: bytes 6240..6244
        let selected_offset = 64 + FUSED_OUTPUT * 4 + NUM_EXPERTS * 4;
        let selected_bytes = &bytes[selected_offset..selected_offset + TOP_K_EXPERTS];

        // Aux data reference (if present)
        let aux_data_offset = u64::from_be(header_ref.aux_data_offset) as usize;
        let aux_data_length = u64::from_be(header_ref.aux_data_length) as usize;
        let aux_data = if aux_data_offset > 0 && aux_data_length > 0 {
            if bytes.len() < aux_data_offset + aux_data_length {
                return Err(DecodeError::BufferTooShort {
                    needed: aux_data_offset + aux_data_length,
                    available: bytes.len(),
                });
            }
            Some(&bytes[aux_data_offset..aux_data_offset + aux_data_length])
        } else {
            None
        };

        Ok(FusedEmbeddingRef {
            header: header_ref,
            vector_bytes,
            weights_bytes,
            selected_bytes,
            aux_data,
        })
    }
}
