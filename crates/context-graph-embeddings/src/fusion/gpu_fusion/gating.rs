//! GPU-accelerated Gating Network for FuseMoE routing.
//!
//! Routes 8320D concatenated embeddings to 8 experts using
//! temperature-scaled softmax with optional Laplace smoothing.

#[cfg(feature = "candle")]
use candle_core::{Device, Tensor, D};
#[cfg(feature = "candle")]
use candle_nn;

use crate::config::FusionConfig;
use crate::error::{EmbeddingError, EmbeddingResult};
use crate::types::dimensions::TOTAL_CONCATENATED;

use super::layer_norm::GpuLayerNorm;
use super::linear::GpuLinear;

/// GPU-accelerated Gating Network for FuseMoE routing.
///
/// Routes 8320D concatenated embeddings to 8 experts using
/// temperature-scaled softmax with optional Laplace smoothing.
///
/// # GPU Acceleration
///
/// Full forward pass on GPU:
/// - LayerNorm: cuBLAS vectorized ops
/// - Linear: cuBLAS GEMM
/// - Softmax: Fused CUDA kernel
///
/// Expected speedup: 60x vs CPU.
#[cfg(feature = "candle")]
#[derive(Debug)]
pub struct GpuGatingNetwork {
    /// Layer normalization for input
    layer_norm: GpuLayerNorm,
    /// Linear projection from input_dim to num_experts
    projection: GpuLinear,
    /// Softmax temperature (lower = sharper)
    temperature: f32,
    /// Laplace smoothing alpha (0 = disabled)
    laplace_alpha: f32,
    /// Number of experts
    num_experts: usize,
    /// Reference to device
    #[allow(dead_code)]
    device: Device,
}

#[cfg(feature = "candle")]
impl GpuGatingNetwork {
    /// Create a new GPU GatingNetwork.
    ///
    /// # Arguments
    ///
    /// * `config` - Fusion configuration
    /// * `device` - CUDA device
    ///
    /// # Errors
    ///
    /// - `EmbeddingError::ConfigError` if configuration is invalid
    /// - `EmbeddingError::GpuError` if GPU allocation fails
    pub fn new(config: &FusionConfig, device: &Device) -> EmbeddingResult<Self> {
        config.validate()?;

        let layer_norm = GpuLayerNorm::new(TOTAL_CONCATENATED, device)?;
        let projection = GpuLinear::new(TOTAL_CONCATENATED, config.num_experts, device)?;

        Ok(Self {
            layer_norm,
            projection,
            temperature: config.temperature,
            laplace_alpha: config.laplace_alpha,
            num_experts: config.num_experts,
            device: device.clone(),
        })
    }

    /// Create GpuGatingNetwork from CPU weights.
    ///
    /// Transfers all parameters to GPU.
    pub fn from_cpu(
        layer_norm_gamma: &[f32],
        layer_norm_beta: &[f32],
        projection_weights: &[f32],
        projection_bias: &[f32],
        config: &FusionConfig,
        device: &Device,
    ) -> EmbeddingResult<Self> {
        let layer_norm = GpuLayerNorm::from_cpu(layer_norm_gamma, layer_norm_beta, device)?;
        let projection = GpuLinear::from_cpu(
            TOTAL_CONCATENATED,
            config.num_experts,
            projection_weights,
            projection_bias,
            device,
        )?;

        Ok(Self {
            layer_norm,
            projection,
            temperature: config.temperature,
            laplace_alpha: config.laplace_alpha,
            num_experts: config.num_experts,
            device: device.clone(),
        })
    }

    /// Get the number of experts.
    #[inline]
    #[must_use]
    pub fn num_experts(&self) -> usize {
        self.num_experts
    }

    /// Get the input dimension.
    #[inline]
    #[must_use]
    pub fn input_dim(&self) -> usize {
        self.layer_norm.dim()
    }

    /// Forward pass through gating network.
    ///
    /// Returns expert probabilities.
    ///
    /// # Arguments
    ///
    /// * `input` - Tensor of shape [batch_size, 8320]
    ///
    /// # Returns
    ///
    /// Probabilities tensor of shape [batch_size, num_experts].
    pub fn forward(&self, input: &Tensor) -> EmbeddingResult<Tensor> {
        // Step 1: Layer normalization
        let normalized = self.layer_norm.forward(input)?;

        // Step 2: Linear projection to logits
        let logits = self.projection.forward(&normalized)?;

        // Step 3: Temperature-scaled softmax
        let probs = self.softmax_with_temperature(&logits)?;

        // Step 4: Laplace smoothing (if enabled)
        if self.laplace_alpha > 0.0 {
            self.apply_laplace_smoothing(&probs)
        } else {
            Ok(probs)
        }
    }

    /// Forward pass with top-k selection.
    ///
    /// Returns (indices, weights) for top-k experts.
    ///
    /// # Arguments
    ///
    /// * `input` - Tensor of shape [batch_size, 8320]
    /// * `top_k` - Number of experts to select
    ///
    /// # Returns
    ///
    /// Tuple of:
    /// - `indices`: Tensor of shape [batch_size, top_k]
    /// - `weights`: Tensor of shape [batch_size, top_k] (renormalized)
    pub fn forward_topk(
        &self,
        input: &Tensor,
        top_k: usize,
    ) -> EmbeddingResult<(Tensor, Tensor)> {
        if top_k > self.num_experts {
            return Err(EmbeddingError::ConfigError {
                message: format!(
                    "top_k ({}) cannot exceed num_experts ({})",
                    top_k, self.num_experts
                ),
            });
        }

        let probs = self.forward(input)?;

        // GPU top-k selection
        let (values, indices) = probs
            .sort_last_dim(false) // descending
            .map_err(|e| EmbeddingError::GpuError {
                message: format!("Sort failed: {}", e),
            })?;

        // Take top-k
        let topk_values = values
            .narrow(D::Minus1, 0, top_k)
            .map_err(|e| EmbeddingError::GpuError {
                message: format!("Narrow values failed: {}", e),
            })?;

        let topk_indices = indices
            .narrow(D::Minus1, 0, top_k)
            .map_err(|e| EmbeddingError::GpuError {
                message: format!("Narrow indices failed: {}", e),
            })?;

        // Renormalize weights to sum to 1
        let weight_sum = topk_values
            .sum_keepdim(D::Minus1)
            .map_err(|e| EmbeddingError::GpuError {
                message: format!("Sum failed: {}", e),
            })?;

        let normalized_weights = topk_values
            .broadcast_div(&weight_sum)
            .map_err(|e| EmbeddingError::GpuError {
                message: format!("Weight normalization failed: {}", e),
            })?;

        Ok((topk_indices, normalized_weights))
    }

    /// Temperature-scaled softmax on GPU.
    fn softmax_with_temperature(&self, logits: &Tensor) -> EmbeddingResult<Tensor> {
        let scaled = (logits / self.temperature as f64).map_err(|e| {
            EmbeddingError::GpuError {
                message: format!("Temperature scaling failed: {}", e),
            }
        })?;

        candle_nn::ops::softmax(&scaled, D::Minus1).map_err(|e| EmbeddingError::GpuError {
            message: format!("Softmax failed: {}", e),
        })
    }

    /// Apply Laplace smoothing on GPU.
    fn apply_laplace_smoothing(&self, probs: &Tensor) -> EmbeddingResult<Tensor> {
        let alpha = self.laplace_alpha as f64;
        let k = self.num_experts as f64;
        let denominator = 1.0 + alpha * k;

        let smoothed = (probs + alpha).map_err(|e| EmbeddingError::GpuError {
            message: format!("Alpha addition failed: {}", e),
        })?;

        (smoothed / denominator).map_err(|e| EmbeddingError::GpuError {
            message: format!("Smoothing division failed: {}", e),
        })
    }

    /// Parameter count for the gating network.
    ///
    /// Includes LayerNorm and Linear projection parameters.
    #[must_use]
    pub fn parameter_count(&self) -> usize {
        self.layer_norm.parameter_count() + self.projection.parameter_count()
    }
}
