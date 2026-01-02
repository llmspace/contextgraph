//! GPU-accelerated Layer Normalization.
//!
//! Normalizes each sample in a batch to mean=0, var=1,
//! then applies learned scale (gamma) and shift (beta).

#[cfg(feature = "candle")]
use candle_core::{DType, Device, Tensor, D};

use crate::error::{EmbeddingError, EmbeddingResult};

/// GPU-accelerated Layer Normalization.
///
/// Normalizes each sample in a batch to mean=0, var=1,
/// then applies learned scale (gamma) and shift (beta).
///
/// # Formula
///
/// ```text
/// y = gamma * (x - mean) / sqrt(var + eps) + beta
/// ```
///
/// # GPU Acceleration
///
/// Uses cuBLAS for vectorized mean/variance computation.
/// Expected speedup: 50x vs CPU for 8320D vectors.
#[cfg(feature = "candle")]
#[derive(Debug)]
pub struct GpuLayerNorm {
    /// Scale parameter (gamma) - shape: [dim]
    gamma: Tensor,
    /// Shift parameter (beta) - shape: [dim]
    beta: Tensor,
    /// Numerical stability constant
    eps: f64,
    /// Expected input dimension
    dim: usize,
}

#[cfg(feature = "candle")]
impl GpuLayerNorm {
    /// Create a new GPU LayerNorm.
    ///
    /// # Arguments
    ///
    /// * `dim` - Input/output dimension (must be > 0)
    /// * `device` - CUDA device for tensor allocation
    ///
    /// # Errors
    ///
    /// - `EmbeddingError::InvalidDimension` if dim == 0
    /// - `EmbeddingError::GpuError` if tensor allocation fails
    pub fn new(dim: usize, device: &Device) -> EmbeddingResult<Self> {
        if dim == 0 {
            return Err(EmbeddingError::InvalidDimension {
                expected: 1,
                actual: 0,
            });
        }

        // Initialize gamma=1.0, beta=0.0 on GPU
        let gamma = Tensor::ones((dim,), DType::F32, device).map_err(|e| {
            EmbeddingError::GpuError {
                message: format!("Failed to allocate gamma tensor: {}", e),
            }
        })?;

        let beta = Tensor::zeros((dim,), DType::F32, device).map_err(|e| {
            EmbeddingError::GpuError {
                message: format!("Failed to allocate beta tensor: {}", e),
            }
        })?;

        Ok(Self {
            gamma,
            beta,
            eps: 1e-5,
            dim,
        })
    }

    /// Create GpuLayerNorm from CPU LayerNorm weights.
    ///
    /// Transfers gamma and beta parameters to GPU.
    ///
    /// # Arguments
    ///
    /// * `gamma` - Scale parameters from CPU
    /// * `beta` - Shift parameters from CPU
    /// * `device` - CUDA device
    ///
    /// # Errors
    ///
    /// - `EmbeddingError::GpuError` if transfer fails
    pub fn from_cpu(
        gamma: &[f32],
        beta: &[f32],
        device: &Device,
    ) -> EmbeddingResult<Self> {
        if gamma.len() != beta.len() {
            return Err(EmbeddingError::DimensionMismatch {
                expected: gamma.len(),
                got: beta.len(),
            });
        }

        let dim = gamma.len();
        if dim == 0 {
            return Err(EmbeddingError::InvalidDimension {
                expected: 1,
                actual: 0,
            });
        }

        let gamma_tensor = Tensor::from_slice(gamma, (dim,), device).map_err(|e| {
            EmbeddingError::GpuError {
                message: format!("Failed to transfer gamma to GPU: {}", e),
            }
        })?;

        let beta_tensor = Tensor::from_slice(beta, (dim,), device).map_err(|e| {
            EmbeddingError::GpuError {
                message: format!("Failed to transfer beta to GPU: {}", e),
            }
        })?;

        Ok(Self {
            gamma: gamma_tensor,
            beta: beta_tensor,
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

    /// Forward pass through GPU layer normalization.
    ///
    /// # Arguments
    ///
    /// * `input` - Tensor of shape [batch_size, dim]
    ///
    /// # Returns
    ///
    /// Normalized tensor of same shape.
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
                    "GpuLayerNorm expects 2D input [batch, dim], got {:?}",
                    input_shape
                ),
            });
        }

        if input_shape[1] != self.dim {
            return Err(EmbeddingError::InvalidDimension {
                expected: self.dim,
                actual: input_shape[1],
            });
        }

        // Compute mean along last dimension: [batch, 1]
        let mean = input
            .mean_keepdim(D::Minus1)
            .map_err(|e| EmbeddingError::GpuError {
                message: format!("Mean computation failed: {}", e),
            })?;

        // Compute variance: E[(x - mean)^2]
        let centered = input
            .broadcast_sub(&mean)
            .map_err(|e| EmbeddingError::GpuError {
                message: format!("Centering failed: {}", e),
            })?;

        let variance = centered
            .sqr()
            .map_err(|e| EmbeddingError::GpuError {
                message: format!("Square failed: {}", e),
            })?
            .mean_keepdim(D::Minus1)
            .map_err(|e| EmbeddingError::GpuError {
                message: format!("Variance computation failed: {}", e),
            })?;

        // Normalize: (x - mean) / sqrt(var + eps)
        let std_inv = (variance + self.eps)
            .map_err(|e| EmbeddingError::GpuError {
                message: format!("Epsilon addition failed: {}", e),
            })?
            .sqrt()
            .map_err(|e| EmbeddingError::GpuError {
                message: format!("Sqrt failed: {}", e),
            })?;

        let normalized = centered
            .broadcast_div(&std_inv)
            .map_err(|e| EmbeddingError::GpuError {
                message: format!("Normalization division failed: {}", e),
            })?;

        // Apply scale and shift: gamma * normalized + beta
        let scaled = normalized
            .broadcast_mul(&self.gamma)
            .map_err(|e| EmbeddingError::GpuError {
                message: format!("Gamma multiplication failed: {}", e),
            })?;

        scaled
            .broadcast_add(&self.beta)
            .map_err(|e| EmbeddingError::GpuError {
                message: format!("Beta addition failed: {}", e),
            })
    }

    /// Parameter count for LayerNorm (gamma + beta).
    #[must_use]
    pub fn parameter_count(&self) -> usize {
        self.dim * 2 // gamma and beta
    }
}
