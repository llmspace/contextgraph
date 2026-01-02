//! FuseMoE fusion layer configuration.
//!
//! Controls the Mixture-of-Experts fusion that combines 12 model outputs
//! into a unified 1536-dimensional embedding.

use serde::{Deserialize, Serialize};

use crate::error::{EmbeddingError, EmbeddingResult};

// ============================================================================
// DEFAULT FUNCTIONS
// ============================================================================

fn default_num_experts() -> usize {
    8
}

fn default_top_k() -> usize {
    4 // FIXED: was 2, constitution.yaml says 4
}

fn default_output_dim() -> usize {
    1536
}

fn default_expert_hidden_dim() -> usize {
    4096
}

fn default_load_balance_coef() -> f32 {
    0.01
}

fn default_capacity_factor() -> f32 {
    1.25
}

fn default_temperature() -> f32 {
    1.0
}

fn default_noise_std() -> f32 {
    0.0
}

fn default_laplace_alpha() -> f32 {
    0.01
}

// ============================================================================
// FUSION CONFIG
// ============================================================================

/// Configuration for FuseMoE fusion layer.
///
/// Controls the Mixture-of-Experts fusion that combines 12 model outputs
/// into a unified 1536-dimensional embedding.
///
/// # Architecture
/// ```text
/// Input: Concatenated embeddings (8320D)
///        |
///        v
///   [Gating Network] --> Expert weights (8 values, temperature-scaled softmax)
///        |
///        v
///   [Top-4 Selection] --> Select 4 experts (Serotonin modulates: range [2,8])
///        |
///        v
///   [Expert Networks] --> Each: 8320 -> 4096 -> 4096 -> 1536
///        |
///        v
///   [Weighted Sum] --> Final 1536D embedding
/// ```
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FusionConfig {
    /// Number of expert networks in MoE.
    /// Constitution spec: 8 experts
    /// Default: 8
    #[serde(default = "default_num_experts")]
    pub num_experts: usize,

    /// Number of experts to activate per input (top-k routing).
    /// Constitution spec: top_k = 4 (NOT 2)
    /// Neuromodulation range: [2, 8] (Serotonin control)
    /// Default: 4
    #[serde(default = "default_top_k")]
    pub top_k: usize,

    /// Output embedding dimension after fusion.
    /// Constitution spec: 1536D (OpenAI ada-002 compatible)
    /// Default: 1536
    #[serde(default = "default_output_dim")]
    pub output_dim: usize,

    /// Hidden dimension in expert FFN layers.
    /// Architecture: input(8320) -> hidden(4096) -> hidden(4096) -> output(1536)
    /// Required by: M03-L21 (Expert Networks), M03-L30 (Grouped GEMM)
    /// Default: 4096
    #[serde(default = "default_expert_hidden_dim")]
    pub expert_hidden_dim: usize,

    /// Load balance loss coefficient.
    /// Penalizes uneven expert utilization during training.
    /// Set to 0.0 to disable, typical range: [0.01, 0.1]
    /// Default: 0.01
    #[serde(default = "default_load_balance_coef")]
    pub load_balance_coef: f32,

    /// Capacity factor for expert buffers.
    /// 1.25 = 25% overhead above average load.
    /// Must be >= 1.0 (no underprovisioning)
    /// Default: 1.25
    #[serde(default = "default_capacity_factor")]
    pub capacity_factor: f32,

    /// Temperature for gating network softmax.
    /// Lower = sharper expert selection, Higher = more uniform distribution
    /// Range: (0, inf), typical: [0.1, 2.0]
    /// Neuromodulation: Noradrenaline modulates attention.temp in range [0.5, 2.0]
    /// Default: 1.0
    #[serde(default = "default_temperature")]
    pub temperature: f32,

    /// Noise standard deviation for exploration (training only).
    /// Gaussian noise added to gating logits before softmax.
    /// Helps prevent expert collapse during training.
    /// Set to 0.0 for inference (deterministic).
    /// Default: 0.0
    #[serde(default = "default_noise_std")]
    pub noise_std: f32,

    /// Laplace smoothing alpha for stable routing.
    /// Formula: (p + alpha) / (1 + alpha * K)
    /// Prevents zero probabilities in gating.
    /// Default: 0.01
    #[serde(default = "default_laplace_alpha")]
    pub laplace_alpha: f32,
}

impl Default for FusionConfig {
    fn default() -> Self {
        Self {
            num_experts: default_num_experts(),
            top_k: default_top_k(),
            output_dim: default_output_dim(),
            expert_hidden_dim: default_expert_hidden_dim(),
            load_balance_coef: default_load_balance_coef(),
            capacity_factor: default_capacity_factor(),
            temperature: default_temperature(),
            noise_std: default_noise_std(),
            laplace_alpha: default_laplace_alpha(),
        }
    }
}

impl FusionConfig {
    /// Validate fusion configuration values.
    ///
    /// # Errors
    /// Returns `EmbeddingError::ConfigError` if:
    /// - num_experts == 0
    /// - top_k == 0 or top_k > num_experts
    /// - output_dim == 0
    /// - expert_hidden_dim == 0
    /// - temperature <= 0 or is NaN
    /// - capacity_factor < 1.0 or is NaN
    /// - laplace_alpha < 0 or is NaN
    /// - noise_std < 0 or is NaN
    /// - load_balance_coef < 0 or is NaN
    pub fn validate(&self) -> EmbeddingResult<()> {
        if self.num_experts == 0 {
            return Err(EmbeddingError::ConfigError {
                message: "num_experts must be > 0".to_string(),
            });
        }
        if self.top_k == 0 || self.top_k > self.num_experts {
            return Err(EmbeddingError::ConfigError {
                message: format!(
                    "top_k must be in [1, {}], got {}",
                    self.num_experts, self.top_k
                ),
            });
        }
        if self.output_dim == 0 {
            return Err(EmbeddingError::ConfigError {
                message: "output_dim must be > 0".to_string(),
            });
        }
        if self.expert_hidden_dim == 0 {
            return Err(EmbeddingError::ConfigError {
                message: "expert_hidden_dim must be > 0".to_string(),
            });
        }
        if self.temperature <= 0.0 || self.temperature.is_nan() {
            return Err(EmbeddingError::ConfigError {
                message: "temperature must be > 0 and not NaN".to_string(),
            });
        }
        if self.capacity_factor < 1.0 || self.capacity_factor.is_nan() {
            return Err(EmbeddingError::ConfigError {
                message: "capacity_factor must be >= 1.0 and not NaN".to_string(),
            });
        }
        if self.laplace_alpha < 0.0 || self.laplace_alpha.is_nan() {
            return Err(EmbeddingError::ConfigError {
                message: "laplace_alpha must be >= 0 and not NaN".to_string(),
            });
        }
        if self.noise_std < 0.0 || self.noise_std.is_nan() {
            return Err(EmbeddingError::ConfigError {
                message: "noise_std must be >= 0 and not NaN".to_string(),
            });
        }
        if self.load_balance_coef < 0.0 || self.load_balance_coef.is_nan() {
            return Err(EmbeddingError::ConfigError {
                message: "load_balance_coef must be >= 0 and not NaN".to_string(),
            });
        }
        Ok(())
    }

    /// Create inference configuration (deterministic, no noise).
    ///
    /// Returns config with:
    /// - noise_std = 0.0 (no exploration noise)
    /// - load_balance_coef = 0.0 (no load balancing loss)
    ///
    /// Use this for production inference where determinism is required.
    pub fn for_inference() -> Self {
        Self {
            noise_std: 0.0,
            load_balance_coef: 0.0,
            ..Default::default()
        }
    }

    /// Create training configuration (with exploration noise).
    ///
    /// Returns config with:
    /// - noise_std = 0.1 (Gaussian noise for exploration)
    /// - load_balance_coef = 0.01 (auxiliary loss for load balancing)
    ///
    /// Use this for training to prevent expert collapse.
    pub fn for_training() -> Self {
        Self {
            noise_std: 0.1,
            load_balance_coef: 0.01,
            ..Default::default()
        }
    }

    /// Check if this config is for inference mode.
    pub fn is_inference_mode(&self) -> bool {
        self.noise_std == 0.0
    }
}
