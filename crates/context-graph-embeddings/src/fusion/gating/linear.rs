//! Linear projection layer (fully connected / dense).
//!
//! Transforms input from `in_features` to `out_features` dimensions
//! using learned weights and optional bias.

use crate::error::{EmbeddingError, EmbeddingResult};
use rand::Rng;

/// Linear projection layer (fully connected / dense).
///
/// Transforms input from `in_features` to `out_features` dimensions
/// using learned weights and optional bias.
///
/// # Formula
///
/// ```text
/// y = x @ W^T + b
/// ```
///
/// where:
/// - `x`: Input tensor [batch_size, in_features]
/// - `W`: Weight matrix [out_features, in_features]
/// - `b`: Bias vector [out_features]
/// - `y`: Output tensor [batch_size, out_features]
#[derive(Debug, Clone)]
pub struct Linear {
    /// Weight matrix (row-major): [out_features, in_features]
    weights: Vec<f32>,
    /// Bias vector: [out_features]
    bias: Vec<f32>,
    /// Input dimension
    in_features: usize,
    /// Output dimension
    out_features: usize,
}

impl Linear {
    /// Create a new Linear layer with Xavier initialization.
    ///
    /// Weights are initialized using Xavier uniform distribution:
    /// `U(-sqrt(6/(in+out)), sqrt(6/(in+out)))`
    ///
    /// Bias is initialized to zero.
    ///
    /// # Arguments
    ///
    /// * `in_features` - Input dimension (must be > 0)
    /// * `out_features` - Output dimension (must be > 0)
    ///
    /// # Errors
    ///
    /// Returns `EmbeddingError::InvalidDimension` if either dimension is 0.
    ///
    /// # Example
    ///
    /// ```rust
    /// use context_graph_embeddings::fusion::Linear;
    ///
    /// let linear = Linear::new(8320, 8).unwrap();
    /// assert_eq!(linear.in_features(), 8320);
    /// assert_eq!(linear.out_features(), 8);
    /// ```
    pub fn new(in_features: usize, out_features: usize) -> EmbeddingResult<Self> {
        if in_features == 0 {
            return Err(EmbeddingError::InvalidDimension {
                expected: 1,
                actual: 0,
            });
        }
        if out_features == 0 {
            return Err(EmbeddingError::InvalidDimension {
                expected: 1,
                actual: 0,
            });
        }

        // Xavier initialization
        let mut rng = rand::thread_rng();
        let limit = (6.0 / (in_features + out_features) as f64).sqrt();

        let weights: Vec<f32> = (0..(out_features * in_features))
            .map(|_| rng.gen_range((-limit)..limit) as f32)
            .collect();

        let bias = vec![0.0; out_features];

        Ok(Self {
            weights,
            bias,
            in_features,
            out_features,
        })
    }

    /// Create a new Linear layer with provided weights and bias.
    ///
    /// # Arguments
    ///
    /// * `in_features` - Input dimension
    /// * `out_features` - Output dimension
    /// * `weights` - Weight matrix (row-major: [out_features, in_features])
    /// * `bias` - Bias vector: [out_features]
    ///
    /// # Errors
    ///
    /// - `EmbeddingError::InvalidDimension` if weights or bias dimensions don't match.
    pub fn with_weights(
        in_features: usize,
        out_features: usize,
        weights: Vec<f32>,
        bias: Vec<f32>,
    ) -> EmbeddingResult<Self> {
        let expected_weights = out_features * in_features;
        if weights.len() != expected_weights {
            return Err(EmbeddingError::InvalidDimension {
                expected: expected_weights,
                actual: weights.len(),
            });
        }
        if bias.len() != out_features {
            return Err(EmbeddingError::InvalidDimension {
                expected: out_features,
                actual: bias.len(),
            });
        }

        Ok(Self {
            weights,
            bias,
            in_features,
            out_features,
        })
    }

    /// Get input dimension.
    #[inline]
    #[must_use]
    pub fn in_features(&self) -> usize {
        self.in_features
    }

    /// Get output dimension.
    #[inline]
    #[must_use]
    pub fn out_features(&self) -> usize {
        self.out_features
    }

    /// Forward pass through linear layer.
    ///
    /// Computes `y = x @ W^T + b` for each sample in the batch.
    ///
    /// # Arguments
    ///
    /// * `input` - Input tensor of shape [batch_size * in_features]
    /// * `batch_size` - Number of samples in the batch
    ///
    /// # Returns
    ///
    /// Output tensor of shape [batch_size * out_features].
    ///
    /// # Errors
    ///
    /// - `EmbeddingError::EmptyInput` if batch_size == 0
    /// - `EmbeddingError::InvalidDimension` if input length != batch_size * in_features
    /// - `EmbeddingError::InvalidValue` if input contains NaN
    ///
    /// # Example
    ///
    /// ```rust
    /// use context_graph_embeddings::fusion::Linear;
    ///
    /// let linear = Linear::new(4, 2).unwrap();
    /// let input = vec![1.0, 2.0, 3.0, 4.0]; // batch_size=1
    /// let output = linear.forward(&input, 1).unwrap();
    /// assert_eq!(output.len(), 2);
    /// ```
    pub fn forward(&self, input: &[f32], batch_size: usize) -> EmbeddingResult<Vec<f32>> {
        if batch_size == 0 {
            return Err(EmbeddingError::EmptyInput);
        }

        let expected_len = batch_size * self.in_features;
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

        let mut output = vec![0.0f32; batch_size * self.out_features];

        // Matrix multiplication: y = x @ W^T + b
        for b in 0..batch_size {
            let input_offset = b * self.in_features;
            let output_offset = b * self.out_features;

            for o in 0..self.out_features {
                let mut sum = self.bias[o];
                let weight_offset = o * self.in_features;

                for i in 0..self.in_features {
                    sum += input[input_offset + i] * self.weights[weight_offset + i];
                }

                output[output_offset + o] = sum;
            }
        }

        Ok(output)
    }

    /// Get a reference to the weight matrix.
    #[inline]
    #[must_use]
    pub fn weights(&self) -> &[f32] {
        &self.weights
    }

    /// Get a reference to the bias vector.
    #[inline]
    #[must_use]
    pub fn bias(&self) -> &[f32] {
        &self.bias
    }
}
