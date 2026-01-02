//! Forward pass implementations for the gating network.
//!
//! Contains forward inference methods and top-K selection.

use crate::error::{EmbeddingError, EmbeddingResult};

use super::super::softmax::{
    add_gaussian_noise, apply_laplace_smoothing, softmax_with_temperature,
};
use super::core::GatingNetwork;

impl GatingNetwork {
    /// Forward pass through gating network.
    ///
    /// Computes expert routing probabilities for each sample in the batch.
    ///
    /// # Processing Steps
    ///
    /// 1. Layer normalize input
    /// 2. Linear projection to logits
    /// 3. Temperature scaling (logits / temperature)
    /// 4. Softmax to probabilities
    /// 5. Laplace smoothing (if alpha > 0)
    ///
    /// # Arguments
    ///
    /// * `input` - Concatenated embeddings [batch_size * input_dim]
    /// * `batch_size` - Number of samples in batch
    ///
    /// # Returns
    ///
    /// Expert probabilities [batch_size * num_experts], each row sums to 1.0.
    ///
    /// # Errors
    ///
    /// - `EmbeddingError::EmptyInput` if batch_size == 0
    /// - `EmbeddingError::InvalidDimension` if input dimensions are wrong
    /// - `EmbeddingError::InvalidValue` if input contains NaN
    pub fn forward(&self, input: &[f32], batch_size: usize) -> EmbeddingResult<Vec<f32>> {
        if batch_size == 0 {
            return Err(EmbeddingError::EmptyInput);
        }

        // Step 1: Layer normalization
        let normalized = self.layer_norm.forward(input, batch_size)?;

        // Step 2: Linear projection to logits
        let logits = self.projection.forward(&normalized, batch_size)?;

        // Step 3 & 4: Temperature-scaled softmax
        let mut probs =
            softmax_with_temperature(&logits, batch_size, self.num_experts, self.temperature)?;

        // Step 5: Laplace smoothing (if enabled)
        if self.laplace_alpha > 0.0 {
            apply_laplace_smoothing(&mut probs, batch_size, self.num_experts, self.laplace_alpha);
        }

        Ok(probs)
    }

    /// Forward pass with Gaussian noise for training exploration.
    ///
    /// Adds Gaussian noise to logits before softmax to encourage
    /// exploration of different expert combinations during training.
    ///
    /// # Arguments
    ///
    /// * `input` - Concatenated embeddings [batch_size * input_dim]
    /// * `batch_size` - Number of samples in batch
    ///
    /// # Returns
    ///
    /// Expert probabilities [batch_size * num_experts] with noise-influenced routing.
    ///
    /// # Errors
    ///
    /// Same as `forward()`.
    pub fn forward_with_noise(
        &self,
        input: &[f32],
        batch_size: usize,
    ) -> EmbeddingResult<Vec<f32>> {
        if batch_size == 0 {
            return Err(EmbeddingError::EmptyInput);
        }

        // Step 1: Layer normalization
        let normalized = self.layer_norm.forward(input, batch_size)?;

        // Step 2: Linear projection to logits
        let mut logits = self.projection.forward(&normalized, batch_size)?;

        // Step 2.5: Add Gaussian noise (training only)
        if self.noise_std > 0.0 {
            add_gaussian_noise(&mut logits, self.noise_std);
        }

        // Step 3 & 4: Temperature-scaled softmax
        let mut probs =
            softmax_with_temperature(&logits, batch_size, self.num_experts, self.temperature)?;

        // Step 5: Laplace smoothing (if enabled)
        if self.laplace_alpha > 0.0 {
            apply_laplace_smoothing(&mut probs, batch_size, self.num_experts, self.laplace_alpha);
        }

        Ok(probs)
    }

    /// Select top-K experts based on probabilities.
    ///
    /// Returns the indices of the K experts with highest probabilities
    /// for each sample in the batch.
    ///
    /// # Arguments
    ///
    /// * `probs` - Expert probabilities [batch_size * num_experts]
    /// * `batch_size` - Number of samples in batch
    /// * `top_k` - Number of experts to select per sample
    ///
    /// # Returns
    ///
    /// Tuple of:
    /// - `indices`: Selected expert indices [batch_size * top_k]
    /// - `weights`: Normalized weights for selected experts [batch_size * top_k]
    ///
    /// # Errors
    ///
    /// - `EmbeddingError::ConfigError` if top_k > num_experts
    /// - `EmbeddingError::InvalidDimension` if probs dimensions are wrong
    pub fn select_top_k(
        &self,
        probs: &[f32],
        batch_size: usize,
        top_k: usize,
    ) -> EmbeddingResult<(Vec<usize>, Vec<f32>)> {
        super::super::routing::select_top_k(probs, batch_size, self.num_experts, top_k)
    }
}
