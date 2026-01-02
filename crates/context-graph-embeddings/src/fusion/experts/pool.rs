//! Expert pool for managing multiple experts with top-k routing.
//!
//! This module implements the ExpertPool struct which manages a collection
//! of expert networks and provides top-k routing for mixture-of-experts fusion.

use crate::config::FusionConfig;
use crate::error::{EmbeddingError, EmbeddingResult};
use crate::types::dimensions::{FUSED_OUTPUT, NUM_EXPERTS, TOTAL_CONCATENATED};
use tracing::{debug, warn};

use super::{Activation, Expert};

/// Pool of all 8 expert networks with top-k routing.
///
/// Manages the collection of experts and provides the core `forward_topk`
/// method that computes weighted combinations of expert outputs.
///
/// # Fields
///
/// - `experts`: Array of 8 experts (always length NUM_EXPERTS)
/// - `input_dim`: Input dimension (8320)
/// - `hidden_dim`: Hidden dimension (4096)
/// - `output_dim`: Output dimension (1536)
///
/// # Example
///
/// ```rust
/// use context_graph_embeddings::fusion::experts::ExpertPool;
/// use context_graph_embeddings::config::FusionConfig;
/// use context_graph_embeddings::types::dimensions::{TOTAL_CONCATENATED, FUSED_OUTPUT, TOP_K_EXPERTS};
///
/// let config = FusionConfig::default();
/// let pool = ExpertPool::new(&config).unwrap();
///
/// let input = vec![0.1f32; TOTAL_CONCATENATED];
/// let indices = vec![0, 2, 4, 6];
/// let weights = vec![0.4, 0.3, 0.2, 0.1];
///
/// let output = pool.forward_topk(&input, 1, &indices, &weights, TOP_K_EXPERTS).unwrap();
/// assert_eq!(output.len(), FUSED_OUTPUT);
/// ```
#[derive(Debug, Clone)]
pub struct ExpertPool {
    /// Array of 8 experts
    experts: Vec<Expert>,
    /// Input dimension (8320)
    input_dim: usize,
    /// Hidden dimension (4096)
    hidden_dim: usize,
    /// Output dimension (1536)
    output_dim: usize,
}

impl ExpertPool {
    /// Create new expert pool from config.
    ///
    /// # Arguments
    ///
    /// * `config` - FusionConfig with expert_hidden_dim
    ///
    /// # Errors
    ///
    /// Returns error if expert initialization fails.
    ///
    /// # Example
    ///
    /// ```rust
    /// use context_graph_embeddings::fusion::experts::ExpertPool;
    /// use context_graph_embeddings::config::FusionConfig;
    ///
    /// let config = FusionConfig::default();
    /// let pool = ExpertPool::new(&config).unwrap();
    /// assert_eq!(pool.num_experts(), 8);
    /// ```
    pub fn new(config: &FusionConfig) -> EmbeddingResult<Self> {
        let input_dim = TOTAL_CONCATENATED;
        let hidden_dim = config.expert_hidden_dim;
        let output_dim = FUSED_OUTPUT;

        let mut experts = Vec::with_capacity(NUM_EXPERTS);

        for expert_id in 0..NUM_EXPERTS {
            let expert = Expert::new(
                expert_id,
                input_dim,
                hidden_dim,
                output_dim,
                Activation::Gelu,
            )?;
            experts.push(expert);
        }

        debug!(
            num_experts = NUM_EXPERTS,
            input_dim,
            hidden_dim,
            output_dim,
            "Created ExpertPool"
        );

        Ok(Self {
            experts,
            input_dim,
            hidden_dim,
            output_dim,
        })
    }

    /// Forward pass through a single expert by index.
    ///
    /// # Arguments
    ///
    /// * `input` - Input [batch_size * input_dim]
    /// * `batch_size` - Number of samples
    /// * `expert_idx` - Expert index 0..NUM_EXPERTS
    ///
    /// # Errors
    ///
    /// * `EmbeddingError::InvalidExpertIndex` if expert_idx >= NUM_EXPERTS
    /// * `EmbeddingError::EmptyInput` if batch_size is 0
    /// * `EmbeddingError::DimensionMismatch` if input length is wrong
    pub fn forward(
        &self,
        input: &[f32],
        batch_size: usize,
        expert_idx: usize,
    ) -> EmbeddingResult<Vec<f32>> {
        if expert_idx >= NUM_EXPERTS {
            return Err(EmbeddingError::InvalidExpertIndex {
                index: expert_idx,
                max: NUM_EXPERTS,
            });
        }

        self.experts[expert_idx].forward(input, batch_size)
    }

    /// Forward through top-k experts with weighted combination.
    ///
    /// This is the PRIMARY method consumed by FuseMoE Router.
    ///
    /// # Arguments
    ///
    /// * `input` - Concatenated embedding [batch_size * 8320]
    /// * `batch_size` - Number of samples in batch
    /// * `indices` - Expert indices from GatingNetwork.select_top_k() [batch_size * top_k]
    /// * `weights` - Routing weights from GatingNetwork.select_top_k() [batch_size * top_k]
    /// * `top_k` - Number of experts per sample (typically 4)
    ///
    /// # Returns
    ///
    /// * Weighted combination of expert outputs [batch_size * 1536]
    ///
    /// # Formula
    ///
    /// For each sample: output = sum(weights[i] * expert[indices[i]].forward(input))
    ///
    /// # Errors
    ///
    /// * `EmbeddingError::EmptyInput` if batch_size is 0
    /// * `EmbeddingError::DimensionMismatch` if input/indices/weights lengths don't match
    /// * `EmbeddingError::InvalidExpertIndex` if any index >= NUM_EXPERTS
    pub fn forward_topk(
        &self,
        input: &[f32],
        batch_size: usize,
        indices: &[usize],
        weights: &[f32],
        top_k: usize,
    ) -> EmbeddingResult<Vec<f32>> {
        // Validate inputs
        if batch_size == 0 {
            return Err(EmbeddingError::EmptyInput);
        }

        if input.len() != batch_size * self.input_dim {
            return Err(EmbeddingError::DimensionMismatch {
                expected: batch_size * self.input_dim,
                got: input.len(),
            });
        }

        if indices.len() != batch_size * top_k {
            return Err(EmbeddingError::DimensionMismatch {
                expected: batch_size * top_k,
                got: indices.len(),
            });
        }

        if weights.len() != batch_size * top_k {
            return Err(EmbeddingError::DimensionMismatch {
                expected: batch_size * top_k,
                got: weights.len(),
            });
        }

        // Validate all indices
        for &idx in indices {
            if idx >= NUM_EXPERTS {
                return Err(EmbeddingError::InvalidExpertIndex {
                    index: idx,
                    max: NUM_EXPERTS,
                });
            }
        }

        // Log warning if weights don't sum to ~1.0
        for sample_idx in 0..batch_size {
            let weight_start = sample_idx * top_k;
            let weight_end = weight_start + top_k;
            let weight_sum: f32 = weights[weight_start..weight_end].iter().sum();
            if (weight_sum - 1.0).abs() > 0.01 {
                warn!(
                    sample_idx,
                    weight_sum, "Weights do not sum to 1.0 (deviation > 0.01)"
                );
            }
        }

        debug!(
            batch_size,
            top_k,
            input_len = input.len(),
            "ExpertPool forward_topk"
        );

        // Initialize output buffer
        let mut output = vec![0.0f32; batch_size * self.output_dim];

        // Process each sample in batch
        for sample_idx in 0..batch_size {
            let input_start = sample_idx * self.input_dim;
            let input_end = input_start + self.input_dim;
            let sample_input = &input[input_start..input_end];

            let output_start = sample_idx * self.output_dim;

            // Process each selected expert for this sample
            for k in 0..top_k {
                let routing_idx = sample_idx * top_k + k;
                let expert_idx = indices[routing_idx];
                let weight = weights[routing_idx];

                // Forward through single expert (batch_size=1 for this sample)
                let expert_output = self.experts[expert_idx].forward(sample_input, 1)?;

                // Weighted accumulation
                for (j, &val) in expert_output.iter().enumerate() {
                    output[output_start + j] += weight * val;
                }
            }
        }

        Ok(output)
    }

    /// Get number of experts in pool.
    #[inline]
    #[must_use]
    pub fn num_experts(&self) -> usize {
        self.experts.len()
    }

    /// Get total parameter count across all experts.
    ///
    /// For default configuration (8320 -> 4096 -> 1536, 8 experts):
    /// - Per expert: ~40.4M parameters
    /// - Total: ~323M parameters
    #[must_use]
    pub fn total_parameter_count(&self) -> usize {
        self.experts.iter().map(|e| e.parameter_count()).sum()
    }

    /// Get input dimension.
    #[inline]
    #[must_use]
    pub fn input_dim(&self) -> usize {
        self.input_dim
    }

    /// Get hidden dimension.
    #[inline]
    #[must_use]
    pub fn hidden_dim(&self) -> usize {
        self.hidden_dim
    }

    /// Get output dimension.
    #[inline]
    #[must_use]
    pub fn output_dim(&self) -> usize {
        self.output_dim
    }
}
