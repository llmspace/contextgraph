//! Core struct definitions for FusedEmbedding and AuxiliaryEmbeddingData.

use crate::error::{EmbeddingError, EmbeddingResult};
use crate::types::dimensions::{COLBERT_V3_DIM, FUSED_OUTPUT, NUM_EXPERTS, TOP_K_EXPERTS};
use crate::types::ModelId;
use serde::{Deserialize, Serialize};

/// Final 1536D fused embedding from FuseMoE expert fusion.
/// This is the primary embedding representation for similarity search.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct FusedEmbedding {
    /// 1536D fused vector (FUSED_OUTPUT dimension)
    pub vector: Vec<f32>,
    /// Expert weights for all 8 experts (sum to 1.0)
    pub expert_weights: [f32; 8],
    /// Indices of top-K selected experts (0-7 each)
    pub selected_experts: [u8; TOP_K_EXPERTS],
    /// Pipeline latency in microseconds
    pub pipeline_latency_us: u64,
    /// xxHash64 of original content for caching
    pub content_hash: u64,
    /// Optional ColBERT per-token vectors for late interaction
    #[serde(skip_serializing_if = "Option::is_none")]
    pub aux_data: Option<AuxiliaryEmbeddingData>,
}

/// Auxiliary embedding data for ColBERT token-level storage.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct AuxiliaryEmbeddingData {
    /// Model that produced this auxiliary data (must be ColBertV3/LateInteraction)
    pub source_model: ModelId,
    /// Per-token embeddings (128D each)
    pub token_vectors: Vec<Vec<f32>>,
    /// Token count (excludes padding)
    pub num_tokens: usize,
    /// Compressed blob for efficient storage
    pub blob: Option<Vec<u8>>,
}

impl FusedEmbedding {
    /// Fused output dimension (from dimensions.rs)
    pub const DIMENSION: usize = FUSED_OUTPUT;
    /// Number of experts in FuseMoE (from dimensions.rs)
    pub const NUM_EXPERTS: usize = NUM_EXPERTS;
    /// Top-K expert selection (from dimensions.rs)
    pub const TOP_K: usize = TOP_K_EXPERTS;

    /// Create new FusedEmbedding with validation.
    ///
    /// # Arguments
    /// * `vector` - 1536D fused embedding vector
    /// * `expert_weights` - Weights for all 8 experts (must sum to ~1.0)
    /// * `selected_experts` - Indices of top-K selected experts (0-7)
    /// * `pipeline_latency_us` - Pipeline latency in microseconds
    /// * `content_hash` - xxHash64 of original content
    ///
    /// # Returns
    /// * `Ok(FusedEmbedding)` if all validations pass
    /// * `Err(EmbeddingError)` if dimension mismatch or invalid expert indices
    ///
    /// # Panics
    /// This method does NOT panic. Validation errors are returned as Results.
    /// For panic-on-invalid behavior, use `new_unchecked` with prior validation.
    pub fn new(
        vector: Vec<f32>,
        expert_weights: [f32; 8],
        selected_experts: [u8; TOP_K_EXPERTS],
        pipeline_latency_us: u64,
        content_hash: u64,
    ) -> EmbeddingResult<Self> {
        // Validate dimension
        if vector.len() != FUSED_OUTPUT {
            return Err(EmbeddingError::InvalidDimension {
                expected: FUSED_OUTPUT,
                actual: vector.len(),
            });
        }

        // Validate expert indices
        for (i, &idx) in selected_experts.iter().enumerate() {
            if idx as usize >= NUM_EXPERTS {
                return Err(EmbeddingError::FusionError {
                    message: format!("Invalid expert index {}, max is {}", idx, NUM_EXPERTS),
                });
            }
            // Check for duplicate expert selection
            for &other_idx in selected_experts.iter().skip(i + 1) {
                if other_idx == idx {
                    return Err(EmbeddingError::FusionError {
                        message: format!("Duplicate expert index: {}", idx),
                    });
                }
            }
        }

        let embedding = Self {
            vector,
            expert_weights,
            selected_experts,
            pipeline_latency_us,
            content_hash,
            aux_data: None,
        };

        Ok(embedding)
    }

    /// Attach auxiliary ColBERT token vectors.
    ///
    /// # Arguments
    /// * `aux_data` - AuxiliaryEmbeddingData containing token vectors
    ///
    /// # Returns
    /// Self with aux_data attached (builder pattern)
    #[must_use]
    pub fn with_aux_data(mut self, aux_data: AuxiliaryEmbeddingData) -> Self {
        self.aux_data = Some(aux_data);
        self
    }

    /// Check if auxiliary data is present.
    #[inline]
    pub fn has_aux_data(&self) -> bool {
        self.aux_data.is_some()
    }

    /// Get ColBERT token vectors for MaxSim scoring.
    #[inline]
    pub fn get_token_vectors(&self) -> Option<&Vec<Vec<f32>>> {
        self.aux_data.as_ref().map(|a| &a.token_vectors)
    }
}

impl AuxiliaryEmbeddingData {
    /// ColBERT dimension (from dimensions.rs)
    pub const TOKEN_DIM: usize = COLBERT_V3_DIM;

    /// Create new AuxiliaryEmbeddingData.
    ///
    /// # Arguments
    /// * `source_model` - Model that produced this data (should be LateInteraction)
    /// * `token_vectors` - Per-token embeddings (128D each)
    ///
    /// # Panics
    /// Panics if any token vector is not 128D. Use `try_new` for non-panicking version.
    #[must_use]
    pub fn new(source_model: ModelId, token_vectors: Vec<Vec<f32>>) -> Self {
        // Validate token vector dimensions
        for (i, vec) in token_vectors.iter().enumerate() {
            if vec.len() != COLBERT_V3_DIM {
                panic!(
                    "Token vector at index {} has dimension {}, expected {}",
                    i,
                    vec.len(),
                    COLBERT_V3_DIM
                );
            }
        }

        let num_tokens = token_vectors.len();
        Self {
            source_model,
            token_vectors,
            num_tokens,
            blob: None,
        }
    }

    /// Try to create new AuxiliaryEmbeddingData without panicking.
    ///
    /// # Errors
    /// - `EmbeddingError::InvalidDimension` if any token vector is not 128D
    pub fn try_new(
        source_model: ModelId,
        token_vectors: Vec<Vec<f32>>,
    ) -> EmbeddingResult<Self> {
        for vec in token_vectors.iter() {
            if vec.len() != COLBERT_V3_DIM {
                return Err(EmbeddingError::InvalidDimension {
                    expected: COLBERT_V3_DIM,
                    actual: vec.len(),
                });
            }
        }

        let num_tokens = token_vectors.len();
        Ok(Self {
            source_model,
            token_vectors,
            num_tokens,
            blob: None,
        })
    }

    /// Calculate memory size in bytes.
    ///
    /// Includes token vector storage only (not metadata).
    #[inline]
    pub fn memory_size(&self) -> usize {
        self.num_tokens * COLBERT_V3_DIM * std::mem::size_of::<f32>()
    }
}
