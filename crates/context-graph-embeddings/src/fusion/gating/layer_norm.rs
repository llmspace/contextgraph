//! Layer normalization for input stabilization.
//!
//! Normalizes each sample in a batch to have mean=0 and variance=1,
//! then applies learned scale (gamma) and shift (beta) parameters.

use crate::error::{EmbeddingError, EmbeddingResult};

/// Layer normalization for input stabilization.
///
/// Normalizes each sample in a batch to have mean=0 and variance=1,
/// then applies learned scale (gamma) and shift (beta) parameters.
///
/// # Formula
///
/// ```text
/// y = gamma * (x - mean) / sqrt(var + eps) + beta
/// ```
///
/// # Fields
///
/// - `gamma`: Scale parameter (learned, initialized to 1.0)
/// - `beta`: Shift parameter (learned, initialized to 0.0)
/// - `eps`: Numerical stability constant (1e-5)
/// - `dim`: Expected input dimension
#[derive(Debug, Clone)]
pub struct LayerNorm {
    /// Scale parameter (gamma) - shape: [dim]
    gamma: Vec<f32>,
    /// Shift parameter (beta) - shape: [dim]
    beta: Vec<f32>,
    /// Numerical stability constant
    eps: f32,
    /// Expected input dimension
    dim: usize,
}

impl LayerNorm {
    /// Create a new LayerNorm with given dimension.
    ///
    /// Initializes:
    /// - `gamma = 1.0` (no scaling)
    /// - `beta = 0.0` (no shift)
    /// - `eps = 1e-5` (numerical stability)
    ///
    /// # Arguments
    ///
    /// * `dim` - Input/output dimension (must be > 0)
    ///
    /// # Errors
    ///
    /// Returns `EmbeddingError::InvalidDimension` if dim == 0.
    ///
    /// # Example
    ///
    /// ```rust
    /// use context_graph_embeddings::fusion::LayerNorm;
    ///
    /// let norm = LayerNorm::new(8320).unwrap();
    /// assert_eq!(norm.dim(), 8320);
    /// ```
    pub fn new(dim: usize) -> EmbeddingResult<Self> {
        if dim == 0 {
            return Err(EmbeddingError::InvalidDimension {
                expected: 1,
                actual: 0,
            });
        }

        Ok(Self {
            gamma: vec![1.0; dim],
            beta: vec![0.0; dim],
            eps: 1e-5,
            dim,
        })
    }

    /// Get the input dimension.
    #[inline]
    #[must_use]
    pub fn dim(&self) -> usize {
        self.dim
    }

    /// Get the epsilon value.
    #[inline]
    #[must_use]
    pub fn eps(&self) -> f32 {
        self.eps
    }

    /// Forward pass through layer normalization.
    ///
    /// Normalizes each sample in the batch independently.
    ///
    /// # Arguments
    ///
    /// * `input` - Input tensor of shape [batch_size * dim]
    /// * `batch_size` - Number of samples in the batch
    ///
    /// # Returns
    ///
    /// Normalized output tensor of shape [batch_size * dim].
    ///
    /// # Errors
    ///
    /// - `EmbeddingError::EmptyInput` if batch_size == 0
    /// - `EmbeddingError::InvalidDimension` if input length != batch_size * dim
    /// - `EmbeddingError::InvalidValue` if input contains NaN
    ///
    /// # Example
    ///
    /// ```rust
    /// use context_graph_embeddings::fusion::LayerNorm;
    ///
    /// let norm = LayerNorm::new(4).unwrap();
    /// let input = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
    /// let output = norm.forward(&input, 2).unwrap();
    /// assert_eq!(output.len(), 8);
    ///
    /// // Verify normalization (each sample should have mean close to 0)
    /// let sample1_mean: f32 = output[0..4].iter().sum::<f32>() / 4.0;
    /// assert!(sample1_mean.abs() < 1e-5);
    /// ```
    pub fn forward(&self, input: &[f32], batch_size: usize) -> EmbeddingResult<Vec<f32>> {
        if batch_size == 0 {
            return Err(EmbeddingError::EmptyInput);
        }

        let expected_len = batch_size * self.dim;
        if input.len() != expected_len {
            return Err(EmbeddingError::InvalidDimension {
                expected: expected_len,
                actual: input.len(),
            });
        }

        // Check for NaN values
        for (i, &val) in input.iter().enumerate() {
            if val.is_nan() {
                return Err(EmbeddingError::InvalidValue { index: i, value: val });
            }
        }

        let mut output = vec![0.0f32; expected_len];

        for b in 0..batch_size {
            let start = b * self.dim;
            let end = start + self.dim;
            let sample = &input[start..end];

            // Compute mean
            let mean: f32 = sample.iter().sum::<f32>() / self.dim as f32;

            // Compute variance
            let variance: f32 = sample
                .iter()
                .map(|x| (x - mean).powi(2))
                .sum::<f32>()
                / self.dim as f32;

            // Normalize: (x - mean) / sqrt(var + eps)
            let inv_std = 1.0 / (variance + self.eps).sqrt();

            for (i, &x) in sample.iter().enumerate() {
                let normalized = (x - mean) * inv_std;
                // Apply scale and shift
                output[start + i] = self.gamma[i] * normalized + self.beta[i];
            }
        }

        Ok(output)
    }

    /// Set the gamma (scale) parameters.
    ///
    /// # Errors
    ///
    /// Returns `EmbeddingError::InvalidDimension` if gamma.len() != dim.
    pub fn set_gamma(&mut self, gamma: Vec<f32>) -> EmbeddingResult<()> {
        if gamma.len() != self.dim {
            return Err(EmbeddingError::InvalidDimension {
                expected: self.dim,
                actual: gamma.len(),
            });
        }
        self.gamma = gamma;
        Ok(())
    }

    /// Set the beta (shift) parameters.
    ///
    /// # Errors
    ///
    /// Returns `EmbeddingError::InvalidDimension` if beta.len() != dim.
    pub fn set_beta(&mut self, beta: Vec<f32>) -> EmbeddingResult<()> {
        if beta.len() != self.dim {
            return Err(EmbeddingError::InvalidDimension {
                expected: self.dim,
                actual: beta.len(),
            });
        }
        self.beta = beta;
        Ok(())
    }

    /// Get a reference to gamma (scale) parameters.
    #[inline]
    #[must_use]
    pub fn gamma(&self) -> &[f32] {
        &self.gamma
    }

    /// Get a reference to beta (shift) parameters.
    #[inline]
    #[must_use]
    pub fn beta(&self) -> &[f32] {
        &self.beta
    }
}
