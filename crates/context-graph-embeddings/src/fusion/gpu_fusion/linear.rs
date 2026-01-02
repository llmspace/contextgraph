//! GPU-accelerated Linear (Fully Connected) Layer.
//!
//! Computes: y = x @ W^T + b

#[cfg(feature = "candle")]
use candle_core::{DType, Device, Tensor};

use crate::error::{EmbeddingError, EmbeddingResult};

/// GPU-accelerated Linear (Fully Connected) Layer.
///
/// Computes: y = x @ W^T + b
///
/// # GPU Acceleration
///
/// Uses cuBLAS GEMM for matrix multiplication.
/// Expected speedup: 50-100x vs CPU for large matrices.
#[cfg(feature = "candle")]
#[derive(Debug)]
pub struct GpuLinear {
    /// Weight matrix: [out_features, in_features]
    weight: Tensor,
    /// Bias vector: [out_features]
    bias: Tensor,
    /// Input dimension
    in_features: usize,
    /// Output dimension
    out_features: usize,
}

#[cfg(feature = "candle")]
impl GpuLinear {
    /// Create a new GPU Linear layer with Xavier initialization.
    ///
    /// # Arguments
    ///
    /// * `in_features` - Input dimension (must be > 0)
    /// * `out_features` - Output dimension (must be > 0)
    /// * `device` - CUDA device
    ///
    /// # Errors
    ///
    /// - `EmbeddingError::InvalidDimension` if dimensions are invalid
    /// - `EmbeddingError::GpuError` if tensor allocation fails
    pub fn new(
        in_features: usize,
        out_features: usize,
        device: &Device,
    ) -> EmbeddingResult<Self> {
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

        // Xavier initialization: U(-sqrt(6/(in+out)), sqrt(6/(in+out)))
        let limit = (6.0 / (in_features + out_features) as f64).sqrt();

        // Use randn and scale for approximate Xavier
        let weight = Tensor::randn(
            0.0f32,
            (limit * 0.5) as f32, // scale std for Xavier-ish
            (out_features, in_features),
            device,
        )
        .map_err(|e| EmbeddingError::GpuError {
            message: format!("Failed to allocate weight tensor: {}", e),
        })?;

        let bias = Tensor::zeros((out_features,), DType::F32, device).map_err(|e| {
            EmbeddingError::GpuError {
                message: format!("Failed to allocate bias tensor: {}", e),
            }
        })?;

        Ok(Self {
            weight,
            bias,
            in_features,
            out_features,
        })
    }

    /// Create GpuLinear from CPU weights.
    ///
    /// # Arguments
    ///
    /// * `in_features` - Input dimension
    /// * `out_features` - Output dimension
    /// * `weights` - Weight matrix (row-major: [out_features, in_features])
    /// * `bias` - Bias vector: [out_features]
    /// * `device` - CUDA device
    ///
    /// # Errors
    ///
    /// - `EmbeddingError::DimensionMismatch` if dimensions don't match
    /// - `EmbeddingError::GpuError` if transfer fails
    pub fn from_cpu(
        in_features: usize,
        out_features: usize,
        weights: &[f32],
        bias: &[f32],
        device: &Device,
    ) -> EmbeddingResult<Self> {
        let expected_weights = out_features * in_features;
        if weights.len() != expected_weights {
            return Err(EmbeddingError::DimensionMismatch {
                expected: expected_weights,
                got: weights.len(),
            });
        }
        if bias.len() != out_features {
            return Err(EmbeddingError::DimensionMismatch {
                expected: out_features,
                got: bias.len(),
            });
        }

        let weight_tensor = Tensor::from_slice(
            weights,
            (out_features, in_features),
            device,
        )
        .map_err(|e| EmbeddingError::GpuError {
            message: format!("Failed to transfer weights to GPU: {}", e),
        })?;

        let bias_tensor = Tensor::from_slice(bias, (out_features,), device).map_err(|e| {
            EmbeddingError::GpuError {
                message: format!("Failed to transfer bias to GPU: {}", e),
            }
        })?;

        Ok(Self {
            weight: weight_tensor,
            bias: bias_tensor,
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

    /// Forward pass through GPU linear layer.
    ///
    /// Computes: y = x @ W^T + b
    ///
    /// # Arguments
    ///
    /// * `input` - Tensor of shape [batch_size, in_features]
    ///
    /// # Returns
    ///
    /// Output tensor of shape [batch_size, out_features].
    ///
    /// # Errors
    ///
    /// - `EmbeddingError::GpuError` if GPU operation fails
    /// - `EmbeddingError::InvalidDimension` if input shape is wrong
    pub fn forward(&self, input: &Tensor) -> EmbeddingResult<Tensor> {
        let input_shape = input.dims();
        if input_shape.len() != 2 {
            return Err(EmbeddingError::GpuError {
                message: format!(
                    "GpuLinear expects 2D input [batch, in_features], got {:?}",
                    input_shape
                ),
            });
        }

        if input_shape[1] != self.in_features {
            return Err(EmbeddingError::InvalidDimension {
                expected: self.in_features,
                actual: input_shape[1],
            });
        }

        // y = x @ W^T
        // input: [batch, in] @ weight.T: [in, out] -> [batch, out]
        let weight_t = self.weight.t().map_err(|e| EmbeddingError::GpuError {
            message: format!("Weight transpose failed: {}", e),
        })?;

        let matmul_result = input.matmul(&weight_t).map_err(|e| {
            EmbeddingError::GpuError {
                message: format!("Matrix multiplication failed: {}", e),
            }
        })?;

        // Add bias
        matmul_result
            .broadcast_add(&self.bias)
            .map_err(|e| EmbeddingError::GpuError {
                message: format!("Bias addition failed: {}", e),
            })
    }

    /// Parameter count for Linear layer (weight + bias).
    #[must_use]
    pub fn parameter_count(&self) -> usize {
        self.in_features * self.out_features + self.out_features
    }
}
