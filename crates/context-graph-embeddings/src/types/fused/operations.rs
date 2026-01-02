//! Vector operations for FusedEmbedding: normalization, similarity, etc.

use crate::error::{EmbeddingError, EmbeddingResult};

use super::constants::CORE_BINARY_SIZE;
use super::core::{AuxiliaryEmbeddingData, FusedEmbedding};

impl FusedEmbedding {
    /// Normalize vector to unit length in-place.
    ///
    /// Zero vectors remain unchanged to avoid division by zero.
    pub fn normalize(&mut self) {
        let mag = self.magnitude();
        if mag > f32::EPSILON {
            for val in &mut self.vector {
                *val /= mag;
            }
        }
    }

    /// Compute cosine similarity with another FusedEmbedding.
    ///
    /// For best performance, both vectors should be normalized first.
    ///
    /// # Arguments
    /// * `other` - Another FusedEmbedding to compare against
    ///
    /// # Returns
    /// Cosine similarity in range [-1.0, 1.0]
    pub fn cosine_similarity(&self, other: &FusedEmbedding) -> f32 {
        let dot_product: f32 = self
            .vector
            .iter()
            .zip(other.vector.iter())
            .map(|(a, b)| a * b)
            .sum();

        let norm_a = self.magnitude();
        let norm_b = other.magnitude();

        if norm_a < f32::EPSILON || norm_b < f32::EPSILON {
            return 0.0;
        }

        dot_product / (norm_a * norm_b)
    }

    /// Compute magnitude (L2 norm) of vector.
    #[inline]
    pub fn magnitude(&self) -> f32 {
        let sum_squares: f32 = self.vector.iter().map(|x| x * x).sum();
        sum_squares.sqrt()
    }

    /// Check if vector is normalized (magnitude approx 1.0).
    #[inline]
    pub fn is_normalized(&self) -> bool {
        (self.magnitude() - 1.0).abs() < 1e-5
    }

    /// Total memory size in bytes for cache budgeting.
    ///
    /// Returns the base binary size (6200 bytes) plus any auxiliary data size.
    /// This matches the serialized format size for accurate memory tracking.
    #[inline]
    #[must_use]
    pub fn memory_size(&self) -> usize {
        let aux_size = self
            .aux_data
            .as_ref()
            .map(|a| a.memory_size())
            .unwrap_or(0);
        CORE_BINARY_SIZE + aux_size
    }

    /// Compress aux_data token_vectors into blob.
    ///
    /// # Errors
    /// - `EmbeddingError::FusionError` if no aux_data present
    pub fn compress_aux_data(&mut self) -> EmbeddingResult<()> {
        if let Some(ref mut aux) = self.aux_data {
            let blob = aux.to_blob();
            aux.blob = Some(blob);
            Ok(())
        } else {
            Err(EmbeddingError::FusionError {
                message: "No auxiliary data to compress".to_string(),
            })
        }
    }

    /// Decompress aux_data blob into token_vectors.
    ///
    /// # Errors
    /// - `EmbeddingError::FusionError` if no aux_data or no blob present
    pub fn decompress_aux_data(&mut self) -> EmbeddingResult<()> {
        if let Some(ref mut aux) = self.aux_data {
            if let Some(ref blob) = aux.blob {
                let decompressed = AuxiliaryEmbeddingData::from_blob(blob)?;
                aux.token_vectors = decompressed.token_vectors;
                aux.num_tokens = decompressed.num_tokens;
                Ok(())
            } else {
                Err(EmbeddingError::FusionError {
                    message: "No blob to decompress".to_string(),
                })
            }
        } else {
            Err(EmbeddingError::FusionError {
                message: "No auxiliary data present".to_string(),
            })
        }
    }
}
