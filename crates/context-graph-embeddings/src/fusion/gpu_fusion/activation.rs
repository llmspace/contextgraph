//! GPU-accelerated activation functions.

#[cfg(feature = "candle")]
use candle_core::Tensor;

use crate::error::{EmbeddingError, EmbeddingResult};

/// GPU-accelerated activation functions.
#[cfg(feature = "candle")]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum GpuActivation {
    /// GELU activation (Gaussian Error Linear Unit)
    #[default]
    Gelu,
    /// ReLU activation (Rectified Linear Unit)
    Relu,
    /// SiLU activation (Sigmoid Linear Unit)
    Silu,
}

#[cfg(feature = "candle")]
impl GpuActivation {
    /// Apply activation function to tensor.
    ///
    /// # Arguments
    ///
    /// * `tensor` - Input tensor
    ///
    /// # Returns
    ///
    /// Activated tensor of same shape.
    ///
    /// # Errors
    ///
    /// - `EmbeddingError::GpuError` if GPU operation fails
    pub fn forward(&self, tensor: &Tensor) -> EmbeddingResult<Tensor> {
        match self {
            GpuActivation::Gelu => tensor.gelu().map_err(|e| EmbeddingError::GpuError {
                message: format!("GELU activation failed: {}", e),
            }),
            GpuActivation::Relu => tensor.relu().map_err(|e| EmbeddingError::GpuError {
                message: format!("ReLU activation failed: {}", e),
            }),
            GpuActivation::Silu => tensor.silu().map_err(|e| EmbeddingError::GpuError {
                message: format!("SiLU activation failed: {}", e),
            }),
        }
    }
}
