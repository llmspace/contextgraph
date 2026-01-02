//! GPU-accelerated Expert Network.
//!
//! Single expert FFN: input_dim -> hidden_dim -> GELU -> output_dim

#[cfg(feature = "candle")]
use candle_core::{Device, Tensor};

use crate::error::EmbeddingResult;

use super::activation::GpuActivation;
use super::linear::GpuLinear;

/// GPU-accelerated Expert Network.
///
/// Single expert FFN: input_dim -> hidden_dim -> GELU -> output_dim
///
/// # GPU Acceleration
///
/// - Linear layers: cuBLAS GEMM
/// - Activation: Fused CUDA kernel
///
/// Expected speedup: 60x vs CPU.
#[cfg(feature = "candle")]
#[derive(Debug)]
pub struct GpuExpert {
    /// First linear layer: input_dim -> hidden_dim
    input_to_hidden: GpuLinear,
    /// Second linear layer: hidden_dim -> output_dim
    hidden_to_output: GpuLinear,
    /// Activation function
    activation: GpuActivation,
    /// Expert identifier
    expert_id: usize,
}

#[cfg(feature = "candle")]
impl GpuExpert {
    /// Create a new GPU Expert.
    ///
    /// # Arguments
    ///
    /// * `expert_id` - Unique identifier (0..NUM_EXPERTS)
    /// * `input_dim` - Input dimension (8320)
    /// * `hidden_dim` - Hidden layer dimension (4096)
    /// * `output_dim` - Output dimension (1536)
    /// * `device` - CUDA device
    ///
    /// # Errors
    ///
    /// - `EmbeddingError::GpuError` if allocation fails
    pub fn new(
        expert_id: usize,
        input_dim: usize,
        hidden_dim: usize,
        output_dim: usize,
        device: &Device,
    ) -> EmbeddingResult<Self> {
        let input_to_hidden = GpuLinear::new(input_dim, hidden_dim, device)?;
        let hidden_to_output = GpuLinear::new(hidden_dim, output_dim, device)?;

        tracing::debug!(
            expert_id,
            input_dim,
            hidden_dim,
            output_dim,
            "Created GPU Expert network"
        );

        Ok(Self {
            input_to_hidden,
            hidden_to_output,
            activation: GpuActivation::Gelu,
            expert_id,
        })
    }

    /// Create GpuExpert from CPU weights.
    pub fn from_cpu(
        expert_id: usize,
        input_dim: usize,
        hidden_dim: usize,
        output_dim: usize,
        input_to_hidden_weights: &[f32],
        input_to_hidden_bias: &[f32],
        hidden_to_output_weights: &[f32],
        hidden_to_output_bias: &[f32],
        device: &Device,
    ) -> EmbeddingResult<Self> {
        let input_to_hidden = GpuLinear::from_cpu(
            input_dim,
            hidden_dim,
            input_to_hidden_weights,
            input_to_hidden_bias,
            device,
        )?;

        let hidden_to_output = GpuLinear::from_cpu(
            hidden_dim,
            output_dim,
            hidden_to_output_weights,
            hidden_to_output_bias,
            device,
        )?;

        Ok(Self {
            input_to_hidden,
            hidden_to_output,
            activation: GpuActivation::Gelu,
            expert_id,
        })
    }

    /// Get expert identifier.
    #[inline]
    #[must_use]
    pub fn expert_id(&self) -> usize {
        self.expert_id
    }

    /// Forward pass through expert.
    ///
    /// # Arguments
    ///
    /// * `input` - Tensor of shape [batch_size, input_dim]
    ///
    /// # Returns
    ///
    /// Output tensor of shape [batch_size, output_dim].
    pub fn forward(&self, input: &Tensor) -> EmbeddingResult<Tensor> {
        // Step 1: Input -> Hidden
        let hidden = self.input_to_hidden.forward(input)?;

        // Step 2: Apply activation
        let activated = self.activation.forward(&hidden)?;

        // Step 3: Hidden -> Output
        self.hidden_to_output.forward(&activated)
    }

    /// Get input dimension.
    #[inline]
    #[must_use]
    pub fn input_dim(&self) -> usize {
        self.input_to_hidden.in_features()
    }

    /// Get output dimension.
    #[inline]
    #[must_use]
    pub fn output_dim(&self) -> usize {
        self.hidden_to_output.out_features()
    }

    /// Parameter count for the expert.
    #[must_use]
    pub fn parameter_count(&self) -> usize {
        self.input_to_hidden.parameter_count() + self.hidden_to_output.parameter_count()
    }
}
