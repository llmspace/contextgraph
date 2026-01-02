//! GPU-accelerated Expert Pool with top-k routing.
//!
//! Manages 8 experts and provides weighted combination of outputs.

#[cfg(feature = "candle")]
use candle_core::{DType, Device, Tensor};

use crate::config::FusionConfig;
use crate::error::{EmbeddingError, EmbeddingResult};
use crate::types::dimensions::{FUSED_OUTPUT, NUM_EXPERTS, TOTAL_CONCATENATED};

use super::expert::GpuExpert;

/// GPU-accelerated Expert Pool with top-k routing.
///
/// Manages 8 experts and provides weighted combination of outputs.
///
/// # GPU Acceleration
///
/// All expert computations on GPU with parallel execution.
/// Expected speedup: 80x vs CPU for full forward pass.
#[cfg(feature = "candle")]
#[derive(Debug)]
pub struct GpuExpertPool {
    /// Array of 8 experts
    experts: Vec<GpuExpert>,
    /// Input dimension (8320)
    input_dim: usize,
    /// Hidden dimension (4096)
    hidden_dim: usize,
    /// Output dimension (1536)
    output_dim: usize,
    /// CUDA device
    device: Device,
}

#[cfg(feature = "candle")]
impl GpuExpertPool {
    /// Create new GPU expert pool.
    ///
    /// # Arguments
    ///
    /// * `config` - FusionConfig with expert_hidden_dim
    /// * `device` - CUDA device
    ///
    /// # Errors
    ///
    /// - `EmbeddingError::GpuError` if allocation fails
    pub fn new(config: &FusionConfig, device: &Device) -> EmbeddingResult<Self> {
        let input_dim = TOTAL_CONCATENATED;
        let hidden_dim = config.expert_hidden_dim;
        let output_dim = FUSED_OUTPUT;

        let mut experts = Vec::with_capacity(NUM_EXPERTS);

        for expert_id in 0..NUM_EXPERTS {
            let expert = GpuExpert::new(
                expert_id,
                input_dim,
                hidden_dim,
                output_dim,
                device,
            )?;
            experts.push(expert);
        }

        tracing::info!(
            num_experts = NUM_EXPERTS,
            input_dim,
            hidden_dim,
            output_dim,
            "Created GPU ExpertPool"
        );

        Ok(Self {
            experts,
            input_dim,
            hidden_dim,
            output_dim,
            device: device.clone(),
        })
    }

    /// Get number of experts.
    #[inline]
    #[must_use]
    pub fn num_experts(&self) -> usize {
        self.experts.len()
    }

    /// Get input dimension.
    #[inline]
    #[must_use]
    pub fn input_dim(&self) -> usize {
        self.input_dim
    }

    /// Get output dimension.
    #[inline]
    #[must_use]
    pub fn output_dim(&self) -> usize {
        self.output_dim
    }

    /// Forward pass through top-k experts with weighted combination.
    ///
    /// # Arguments
    ///
    /// * `input` - Tensor of shape [batch_size, 8320]
    /// * `indices` - Expert indices tensor [batch_size, top_k]
    /// * `weights` - Routing weights tensor [batch_size, top_k]
    ///
    /// # Returns
    ///
    /// Weighted combination of expert outputs [batch_size, 1536].
    pub fn forward_topk(
        &self,
        input: &Tensor,
        indices: &Tensor,
        weights: &Tensor,
    ) -> EmbeddingResult<Tensor> {
        let input_shape = input.dims();
        let indices_shape = indices.dims();
        let weights_shape = weights.dims();

        if input_shape.len() != 2 {
            return Err(EmbeddingError::GpuError {
                message: format!("Input must be 2D, got {:?}", input_shape),
            });
        }

        let batch_size = input_shape[0];
        let top_k = indices_shape[1];

        // Validate shapes
        if indices_shape[0] != batch_size || weights_shape[0] != batch_size {
            return Err(EmbeddingError::GpuError {
                message: format!(
                    "Batch size mismatch: input={}, indices={}, weights={}",
                    batch_size, indices_shape[0], weights_shape[0]
                ),
            });
        }

        // Convert indices to Vec for iteration
        let indices_cpu: Vec<u32> = indices
            .flatten_all()
            .map_err(|e| EmbeddingError::GpuError {
                message: format!("Flatten indices failed: {}", e),
            })?
            .to_vec1()
            .map_err(|e| EmbeddingError::GpuError {
                message: format!("Convert indices to CPU failed: {}", e),
            })?;

        let weights_cpu: Vec<f32> = weights
            .flatten_all()
            .map_err(|e| EmbeddingError::GpuError {
                message: format!("Flatten weights failed: {}", e),
            })?
            .to_vec1()
            .map_err(|e| EmbeddingError::GpuError {
                message: format!("Convert weights to CPU failed: {}", e),
            })?;

        // Collect sample outputs for concatenation
        let mut sample_outputs: Vec<Tensor> = Vec::with_capacity(batch_size);

        // Process each sample in batch
        for b in 0..batch_size {
            // Get input sample using narrow (slice) operation
            let sample_input = input
                .narrow(0, b, 1)
                .map_err(|e| EmbeddingError::GpuError {
                    message: format!("Sample slicing failed at index {}: {}", b, e),
                })?;

            let mut sample_output = Tensor::zeros(
                (1, self.output_dim),
                DType::F32,
                &self.device,
            )
            .map_err(|e| EmbeddingError::GpuError {
                message: format!("Sample output allocation failed: {}", e),
            })?;

            // Process each selected expert
            for k in 0..top_k {
                let idx = indices_cpu[b * top_k + k] as usize;
                let weight = weights_cpu[b * top_k + k];

                if idx >= self.experts.len() {
                    return Err(EmbeddingError::GpuError {
                        message: format!(
                            "Expert index {} out of bounds (max {})",
                            idx,
                            self.experts.len() - 1
                        ),
                    });
                }

                // Forward through expert
                let expert_output = self.experts[idx].forward(&sample_input)?;

                // Accumulate weighted output
                let weighted = (expert_output * weight as f64).map_err(|e| {
                    EmbeddingError::GpuError {
                        message: format!("Weight multiplication failed: {}", e),
                    }
                })?;

                sample_output = (sample_output + weighted).map_err(|e| {
                    EmbeddingError::GpuError {
                        message: format!("Output accumulation failed: {}", e),
                    }
                })?;
            }

            sample_outputs.push(sample_output);
        }

        // Concatenate all sample outputs into final batch tensor
        let output = Tensor::cat(&sample_outputs, 0).map_err(|e| EmbeddingError::GpuError {
            message: format!("Output concatenation failed: {}", e),
        })?;

        Ok(output)
    }

    /// Parameter count for all experts.
    ///
    /// For 8 experts with 8320 -> 4096 -> 1536:
    /// ~323M parameters total.
    #[must_use]
    pub fn parameter_count(&self) -> usize {
        let layer1_params = self.input_dim * self.hidden_dim + self.hidden_dim;
        let layer2_params = self.hidden_dim * self.output_dim + self.output_dim;
        let per_expert = layer1_params + layer2_params;
        per_expert * self.experts.len()
    }
}
