//! Binary serialization for FusedEmbedding and AuxiliaryEmbeddingData.

use crate::error::{EmbeddingError, EmbeddingResult};
use crate::types::dimensions::{COLBERT_V3_DIM, FUSED_OUTPUT, TOP_K_EXPERTS};
use crate::types::ModelId;

use super::constants::CORE_BINARY_SIZE;
use super::core::{AuxiliaryEmbeddingData, FusedEmbedding};

impl FusedEmbedding {
    /// Serialize to compact binary format (6200 bytes + aux_data).
    ///
    /// # Binary Layout
    /// | Offset | Size | Field |
    /// |--------|------|-------|
    /// | 0 | 6144 | vector (1536 x f32 LE) |
    /// | 6144 | 32 | expert_weights (8 x f32 LE) |
    /// | 6176 | 4 | selected_experts (4 x u8, TOP_K_EXPERTS) |
    /// | 6180 | 8 | pipeline_latency_us (u64 LE) |
    /// | 6188 | 8 | content_hash (u64 LE) |
    /// | 6196 | 4 | aux_data_len (u32 LE, 0 if None) |
    /// | 6200+ | var | aux_data blob (if present) |
    pub fn to_bytes(&self) -> Vec<u8> {
        let aux_blob = self.aux_data.as_ref().map(|a| a.to_blob());
        let aux_len = aux_blob.as_ref().map(|b| b.len()).unwrap_or(0);
        let mut bytes = Vec::with_capacity(CORE_BINARY_SIZE + aux_len);

        // Vector: 1536 x f32 (6144 bytes)
        for &val in &self.vector {
            bytes.extend_from_slice(&val.to_le_bytes());
        }

        // Expert weights: 8 x f32 (32 bytes)
        for &w in &self.expert_weights {
            bytes.extend_from_slice(&w.to_le_bytes());
        }

        // Selected experts: TOP_K_EXPERTS x u8
        bytes.extend_from_slice(&self.selected_experts);

        // Pipeline latency: u64 (8 bytes)
        bytes.extend_from_slice(&self.pipeline_latency_us.to_le_bytes());

        // Content hash: u64 (8 bytes)
        bytes.extend_from_slice(&self.content_hash.to_le_bytes());

        // Aux data length: u32 (4 bytes)
        bytes.extend_from_slice(&(aux_len as u32).to_le_bytes());

        // Aux data blob (if present)
        if let Some(blob) = aux_blob {
            bytes.extend_from_slice(&blob);
        }

        bytes
    }

    /// Deserialize from binary format.
    ///
    /// # Errors
    /// - `EmbeddingError::SerializationError` if data is truncated or corrupted
    pub fn from_bytes(bytes: &[u8]) -> EmbeddingResult<Self> {
        if bytes.len() < CORE_BINARY_SIZE {
            return Err(EmbeddingError::SerializationError {
                message: format!(
                    "Data too short: expected at least {} bytes, got {}",
                    CORE_BINARY_SIZE,
                    bytes.len()
                ),
            });
        }

        let mut offset = 0;

        // Vector: 1536 x f32
        let mut vector = Vec::with_capacity(FUSED_OUTPUT);
        for _ in 0..FUSED_OUTPUT {
            let val = f32::from_le_bytes([
                bytes[offset],
                bytes[offset + 1],
                bytes[offset + 2],
                bytes[offset + 3],
            ]);
            vector.push(val);
            offset += 4;
        }

        // Expert weights: 8 x f32
        let mut expert_weights = [0.0f32; 8];
        for weight in &mut expert_weights {
            *weight = f32::from_le_bytes([
                bytes[offset],
                bytes[offset + 1],
                bytes[offset + 2],
                bytes[offset + 3],
            ]);
            offset += 4;
        }

        // Selected experts: TOP_K_EXPERTS x u8
        let mut selected_experts = [0u8; TOP_K_EXPERTS];
        for (i, expert) in selected_experts.iter_mut().enumerate() {
            *expert = bytes[offset + i];
        }
        offset += TOP_K_EXPERTS;

        // Pipeline latency: u64
        let pipeline_latency_us = u64::from_le_bytes([
            bytes[offset],
            bytes[offset + 1],
            bytes[offset + 2],
            bytes[offset + 3],
            bytes[offset + 4],
            bytes[offset + 5],
            bytes[offset + 6],
            bytes[offset + 7],
        ]);
        offset += 8;

        // Content hash: u64
        let content_hash = u64::from_le_bytes([
            bytes[offset],
            bytes[offset + 1],
            bytes[offset + 2],
            bytes[offset + 3],
            bytes[offset + 4],
            bytes[offset + 5],
            bytes[offset + 6],
            bytes[offset + 7],
        ]);
        offset += 8;

        // Aux data length: u32
        let aux_len = u32::from_le_bytes([
            bytes[offset],
            bytes[offset + 1],
            bytes[offset + 2],
            bytes[offset + 3],
        ]) as usize;
        offset += 4;

        // Aux data blob (if present)
        let aux_data = if aux_len > 0 {
            if bytes.len() < offset + aux_len {
                return Err(EmbeddingError::SerializationError {
                    message: format!(
                        "Aux data truncated: expected {} bytes at offset {}, got {} total",
                        aux_len,
                        offset,
                        bytes.len()
                    ),
                });
            }
            let blob = &bytes[offset..offset + aux_len];
            Some(AuxiliaryEmbeddingData::from_blob(blob)?)
        } else {
            None
        };

        Ok(Self {
            vector,
            expert_weights,
            selected_experts,
            pipeline_latency_us,
            content_hash,
            aux_data,
        })
    }
}

impl AuxiliaryEmbeddingData {
    /// Serialize to compressed blob.
    ///
    /// Binary format:
    /// - 1 byte: source_model as u8
    /// - 4 bytes: num_tokens as u32 LE
    /// - num_tokens * 128 * 4 bytes: flattened f32 vectors
    pub fn to_blob(&self) -> Vec<u8> {
        let vector_bytes = self.num_tokens * COLBERT_V3_DIM * 4;
        let mut blob = Vec::with_capacity(1 + 4 + vector_bytes);

        // Source model: 1 byte
        blob.push(self.source_model as u8);

        // Num tokens: 4 bytes
        blob.extend_from_slice(&(self.num_tokens as u32).to_le_bytes());

        // Token vectors: flattened f32s
        for vec in &self.token_vectors {
            for &val in vec {
                blob.extend_from_slice(&val.to_le_bytes());
            }
        }

        blob
    }

    /// Deserialize from blob.
    ///
    /// # Errors
    /// - `EmbeddingError::SerializationError` if blob is corrupted or truncated
    pub fn from_blob(blob: &[u8]) -> EmbeddingResult<Self> {
        if blob.len() < 5 {
            return Err(EmbeddingError::SerializationError {
                message: "Blob too short for header".to_string(),
            });
        }

        // Source model: 1 byte
        let source_model = ModelId::try_from(blob[0]).map_err(|e| {
            EmbeddingError::SerializationError {
                message: format!("Invalid source model: {}", e),
            }
        })?;

        // Num tokens: 4 bytes
        let num_tokens =
            u32::from_le_bytes([blob[1], blob[2], blob[3], blob[4]]) as usize;

        let expected_len = 5 + num_tokens * COLBERT_V3_DIM * 4;
        if blob.len() < expected_len {
            return Err(EmbeddingError::SerializationError {
                message: format!(
                    "Blob too short: expected {} bytes for {} tokens, got {}",
                    expected_len, num_tokens, blob.len()
                ),
            });
        }

        // Token vectors
        let mut token_vectors = Vec::with_capacity(num_tokens);
        let mut offset = 5;
        for _ in 0..num_tokens {
            let mut vec = Vec::with_capacity(COLBERT_V3_DIM);
            for _ in 0..COLBERT_V3_DIM {
                let val = f32::from_le_bytes([
                    blob[offset],
                    blob[offset + 1],
                    blob[offset + 2],
                    blob[offset + 3],
                ]);
                vec.push(val);
                offset += 4;
            }
            token_vectors.push(vec);
        }

        Ok(Self {
            source_model,
            token_vectors,
            num_tokens,
            blob: Some(blob.to_vec()),
        })
    }
}
