//! Operations for ConcatenatedEmbedding: concatenation, validation, and slicing.
//!
//! Contains methods that operate on the collected embeddings to produce
//! the final concatenated vector and perform validation.

use crate::error::EmbeddingResult;
use crate::types::dimensions::{self, MODEL_COUNT, TOTAL_CONCATENATED};
use crate::types::ModelId;
use xxhash_rust::xxh64::xxh64;

use super::ConcatenatedEmbedding;

impl ConcatenatedEmbedding {
    /// Builds the concatenated vector from embeddings in model order (E1-E12).
    ///
    /// After this call:
    /// - `self.concatenated.len() == 8320`
    /// - `self.content_hash` is computed via xxHash64
    ///
    /// # Panics
    /// Panics if `is_complete() == false`. All 12 models must be present.
    /// This is fail-fast by design - partial concatenation is not supported.
    ///
    /// # Performance
    /// - Preallocates the full 8320-element vector
    /// - Uses `extend_from_slice` for efficient copying
    /// - Hash computation is O(n) on the final vector
    pub fn concatenate(&mut self) {
        assert!(
            self.is_complete(),
            "Cannot concatenate: {} of 12 models missing. Missing: {:?}",
            MODEL_COUNT - self.filled_count(),
            self.missing_models()
        );

        // Preallocate exact size
        self.concatenated = Vec::with_capacity(TOTAL_CONCATENATED);

        // Concatenate in E1-E12 order
        for model_id in ModelId::all() {
            let embedding = self.embeddings[*model_id as u8 as usize]
                .as_ref()
                .expect("Embedding should exist after is_complete() check");
            self.concatenated.extend_from_slice(&embedding.vector);
        }

        // Compute content hash for caching
        self.content_hash = Self::compute_hash(&self.concatenated);
    }

    /// Returns the total dimension of the concatenated vector.
    ///
    /// Returns `TOTAL_CONCATENATED` (8320) when complete, or the sum of
    /// filled embedding dimensions when incomplete.
    #[must_use]
    pub fn total_dimension(&self) -> usize {
        if !self.concatenated.is_empty() {
            return self.concatenated.len();
        }

        // Sum dimensions of filled embeddings
        self.embeddings
            .iter()
            .filter_map(|e| e.as_ref())
            .map(|e| e.vector.len())
            .sum()
    }

    /// Validates all embeddings against their model requirements.
    ///
    /// Calls `validate()` on each present embedding.
    ///
    /// # Errors
    /// Returns the first validation error encountered.
    /// Does not validate missing embeddings (use `is_complete()` for that).
    pub fn validate(&self) -> EmbeddingResult<()> {
        for embedding in self.embeddings.iter().flatten() {
            embedding.validate()?;
        }
        Ok(())
    }

    /// Computes xxHash64 of concatenated vector bytes.
    ///
    /// This hash is deterministic: same embeddings → same hash.
    /// Used for caching and deduplication.
    ///
    /// # Arguments
    /// * `data` - The f32 slice to hash
    ///
    /// # Returns
    /// The 64-bit hash value
    pub(crate) fn compute_hash(data: &[f32]) -> u64 {
        // Convert f32 slice to bytes for hashing
        // SAFETY: f32 and [u8; 4] have the same size and alignment requirements
        let bytes: &[u8] = unsafe {
            std::slice::from_raw_parts(data.as_ptr() as *const u8, std::mem::size_of_val(data))
        };

        // Use seed 0 for deterministic hashing
        xxh64(bytes, 0)
    }

    /// Returns slice of the concatenated vector at the given model's offset.
    ///
    /// Useful for extracting individual model contributions after concatenation.
    ///
    /// # Arguments
    /// * `model_id` - The model whose slice to extract
    ///
    /// # Returns
    /// - `Some(&[f32])` if concatenated vector exists
    /// - `None` if `concatenate()` hasn't been called
    ///
    /// # Panics
    /// Panics if the concatenated vector is malformed (slice bounds error).
    #[must_use]
    pub fn get_slice(&self, model_id: ModelId) -> Option<&[f32]> {
        if self.concatenated.is_empty() {
            return None;
        }

        let index = model_id as u8 as usize;
        let offset = dimensions::offset_by_index(index);
        let dim = dimensions::projected_dimension_by_index(index);

        Some(&self.concatenated[offset..offset + dim])
    }
}
