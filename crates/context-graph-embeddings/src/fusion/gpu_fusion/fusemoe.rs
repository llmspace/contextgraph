//! GPU-accelerated FuseMoE layer combining gating and experts.
//!
//! Complete fusion layer:
//! 1. GatingNetwork routes input to top-k experts
//! 2. ExpertPool computes weighted expert outputs
//! 3. Returns fused 1536D embedding

#[cfg(feature = "candle")]
use candle_core::{Device, Tensor};

use crate::config::FusionConfig;
use crate::error::EmbeddingResult;

use super::expert_pool::GpuExpertPool;
use super::gating::GpuGatingNetwork;

/// GPU-accelerated FuseMoE layer combining gating and experts.
///
/// Complete fusion layer:
/// 1. GatingNetwork routes input to top-k experts
/// 2. ExpertPool computes weighted expert outputs
/// 3. Returns fused 1536D embedding
#[cfg(feature = "candle")]
#[derive(Debug)]
pub struct GpuFuseMoE {
    /// Gating network for expert routing
    gating: GpuGatingNetwork,
    /// Pool of expert networks
    experts: GpuExpertPool,
    /// Number of experts to use per sample
    top_k: usize,
}

#[cfg(feature = "candle")]
impl GpuFuseMoE {
    /// Create new GPU FuseMoE layer.
    ///
    /// # Arguments
    ///
    /// * `config` - Fusion configuration
    /// * `device` - CUDA device
    ///
    /// # Errors
    ///
    /// - `EmbeddingError::GpuError` if GPU allocation fails
    pub fn new(config: &FusionConfig, device: &Device) -> EmbeddingResult<Self> {
        config.validate()?;

        let gating = GpuGatingNetwork::new(config, device)?;
        let experts = GpuExpertPool::new(config, device)?;

        tracing::info!(
            num_experts = config.num_experts,
            top_k = config.top_k,
            parameter_count = experts.parameter_count(),
            "Created GPU FuseMoE layer"
        );

        Ok(Self {
            gating,
            experts,
            top_k: config.top_k,
        })
    }

    /// Forward pass through full FuseMoE layer.
    ///
    /// # Arguments
    ///
    /// * `input` - Concatenated embeddings tensor [batch_size, 8320]
    ///
    /// # Returns
    ///
    /// Fused embedding tensor [batch_size, 1536].
    ///
    /// # Errors
    ///
    /// - `EmbeddingError::GpuError` if GPU operation fails
    pub fn forward(&self, input: &Tensor) -> EmbeddingResult<Tensor> {
        // Step 1: Get expert routing
        let (indices, weights) = self.gating.forward_topk(input, self.top_k)?;

        // Step 2: Forward through experts with weighted combination
        self.experts.forward_topk(input, &indices, &weights)
    }

    /// Get the gating network (for introspection).
    #[inline]
    #[must_use]
    pub fn gating(&self) -> &GpuGatingNetwork {
        &self.gating
    }

    /// Get the expert pool (for introspection).
    #[inline]
    #[must_use]
    pub fn experts(&self) -> &GpuExpertPool {
        &self.experts
    }

    /// Get top-k value.
    #[inline]
    #[must_use]
    pub fn top_k(&self) -> usize {
        self.top_k
    }

    /// Get input dimension (concatenated embedding size).
    #[inline]
    #[must_use]
    pub fn input_dim(&self) -> usize {
        self.gating.input_dim()
    }

    /// Get output dimension (fused embedding size).
    #[inline]
    #[must_use]
    pub fn output_dim(&self) -> usize {
        self.experts.output_dim()
    }

    /// Total parameter count for the FuseMoE layer.
    ///
    /// Includes:
    /// - Gating network: LayerNorm + Linear
    /// - Expert pool: All expert networks
    #[must_use]
    pub fn parameter_count(&self) -> usize {
        self.gating.parameter_count() + self.experts.parameter_count()
    }
}
