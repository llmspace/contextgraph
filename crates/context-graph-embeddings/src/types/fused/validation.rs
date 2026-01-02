//! Validation methods for FusedEmbedding and AuxiliaryEmbeddingData.

use crate::error::{EmbeddingError, EmbeddingResult};
use crate::types::dimensions::{COLBERT_V3_DIM, NUM_EXPERTS};

use super::constants::WEIGHT_SUM_TOLERANCE;
use super::core::{AuxiliaryEmbeddingData, FusedEmbedding};

impl FusedEmbedding {
    /// Validate embedding state. Returns error if:
    /// - vector contains NaN or Inf
    /// - expert_weights don't sum to ~1.0 (+-0.01)
    /// - selected_experts contain invalid indices (>=8)
    ///
    /// # Errors
    /// - `EmbeddingError::InvalidValue` if NaN or Inf values found
    /// - `EmbeddingError::FusionError` if weights don't sum to ~1.0 or invalid expert indices
    pub fn validate(&self) -> EmbeddingResult<()> {
        // Check for NaN/Inf in vector
        for (idx, &val) in self.vector.iter().enumerate() {
            if val.is_nan() || val.is_infinite() {
                return Err(EmbeddingError::InvalidValue { index: idx, value: val });
            }
        }

        // Check expert weights sum to 1.0 +- tolerance
        let weight_sum: f32 = self.expert_weights.iter().sum();
        if (weight_sum - 1.0).abs() > WEIGHT_SUM_TOLERANCE {
            return Err(EmbeddingError::FusionError {
                message: format!(
                    "Weights sum to {:.6}, expected 1.0 +- {}",
                    weight_sum, WEIGHT_SUM_TOLERANCE
                ),
            });
        }

        // Check expert indices
        for &idx in &self.selected_experts {
            if idx as usize >= NUM_EXPERTS {
                return Err(EmbeddingError::FusionError {
                    message: format!("Invalid expert index {}, max is {}", idx, NUM_EXPERTS),
                });
            }
        }

        Ok(())
    }
}

impl AuxiliaryEmbeddingData {
    /// Validate token vector dimensions.
    ///
    /// # Errors
    /// - `EmbeddingError::InvalidDimension` if any token vector is not 128D
    /// - `EmbeddingError::FusionError` if NaN or Inf values found
    pub fn validate(&self) -> EmbeddingResult<()> {
        for (i, vec) in self.token_vectors.iter().enumerate() {
            if vec.len() != COLBERT_V3_DIM {
                return Err(EmbeddingError::InvalidDimension {
                    expected: COLBERT_V3_DIM,
                    actual: vec.len(),
                });
            }
            // Check for NaN/Inf
            for (j, &val) in vec.iter().enumerate() {
                if val.is_nan() || val.is_infinite() {
                    return Err(EmbeddingError::FusionError {
                        message: format!(
                            "Invalid value {} in token {} at position {}",
                            val, i, j
                        ),
                    });
                }
            }
        }
        Ok(())
    }
}
