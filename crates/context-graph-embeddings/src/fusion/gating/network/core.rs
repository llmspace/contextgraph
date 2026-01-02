//! Core gating network structure and constructors.
//!
//! Contains the `GatingNetwork` struct definition and its constructors.

use crate::config::FusionConfig;
use crate::error::{EmbeddingError, EmbeddingResult};
use crate::types::dimensions::TOTAL_CONCATENATED;

use super::super::{LayerNorm, Linear};

/// Gating network for FuseMoE expert routing.
///
/// Routes the 8320D concatenated embeddings to 8 experts using
/// temperature-scaled softmax with optional Laplace smoothing.
///
/// # Architecture
///
/// 1. **LayerNorm**: Normalize input to mean=0, var=1
/// 2. **Linear**: Project from 8320D to 8D (one logit per expert)
/// 3. **Temperature Scaling**: Scale logits by 1/temperature
/// 4. **Softmax**: Convert to probabilities
/// 5. **Laplace Smoothing** (optional): Prevent zero probabilities
///
/// # Constitution Compliance
///
/// - `num_experts = 8` (constitution.yaml: fuse_moe.num_experts)
/// - `temperature = 1.0` (default, configurable)
/// - `laplace_alpha = 0.01` (constitution.yaml: fuse_moe.laplace_alpha)
///
/// # Example
///
/// ```rust
/// use context_graph_embeddings::fusion::GatingNetwork;
/// use context_graph_embeddings::config::FusionConfig;
///
/// let config = FusionConfig::default();
/// let gating = GatingNetwork::new(&config).unwrap();
///
/// // Forward pass
/// let input = vec![0.0f32; 8320];
/// let probs = gating.forward(&input, 1).unwrap();
///
/// // Verify output
/// assert_eq!(probs.len(), 8);
/// assert!((probs.iter().sum::<f32>() - 1.0).abs() < 1e-5); // Sums to 1
/// assert!(probs.iter().all(|&p| p > 0.0)); // All positive
/// ```
#[derive(Debug, Clone)]
pub struct GatingNetwork {
    /// Layer normalization for input
    pub(super) layer_norm: LayerNorm,
    /// Linear projection from input_dim to num_experts
    pub(super) projection: Linear,
    /// Softmax temperature (lower = sharper)
    pub(super) temperature: f32,
    /// Laplace smoothing alpha (0 = disabled)
    pub(super) laplace_alpha: f32,
    /// Number of experts
    pub(super) num_experts: usize,
    /// Noise standard deviation for training
    pub(super) noise_std: f32,
}

impl GatingNetwork {
    /// Create a new GatingNetwork from FusionConfig.
    ///
    /// # Arguments
    ///
    /// * `config` - Fusion configuration specifying num_experts, temperature, etc.
    ///
    /// # Errors
    ///
    /// - `EmbeddingError::ConfigError` if configuration is invalid
    /// - `EmbeddingError::InvalidDimension` if dimensions are inconsistent
    ///
    /// # Example
    ///
    /// ```rust
    /// use context_graph_embeddings::fusion::GatingNetwork;
    /// use context_graph_embeddings::config::FusionConfig;
    ///
    /// let config = FusionConfig::default();
    /// let gating = GatingNetwork::new(&config).unwrap();
    /// assert_eq!(gating.num_experts(), 8);
    /// ```
    pub fn new(config: &FusionConfig) -> EmbeddingResult<Self> {
        // Validate config
        config.validate()?;

        let layer_norm = LayerNorm::new(TOTAL_CONCATENATED)?;
        let projection = Linear::new(TOTAL_CONCATENATED, config.num_experts)?;

        Ok(Self {
            layer_norm,
            projection,
            temperature: config.temperature,
            laplace_alpha: config.laplace_alpha,
            num_experts: config.num_experts,
            noise_std: config.noise_std,
        })
    }

    /// Create a GatingNetwork with custom input dimension.
    ///
    /// Useful for testing or non-standard configurations.
    ///
    /// # Arguments
    ///
    /// * `input_dim` - Input dimension (typically 8320)
    /// * `config` - Fusion configuration
    ///
    /// # Errors
    ///
    /// - `EmbeddingError::InvalidDimension` if input_dim == 0
    pub fn with_input_dim(input_dim: usize, config: &FusionConfig) -> EmbeddingResult<Self> {
        if input_dim == 0 {
            return Err(EmbeddingError::InvalidDimension {
                expected: 1,
                actual: 0,
            });
        }

        config.validate()?;

        let layer_norm = LayerNorm::new(input_dim)?;
        let projection = Linear::new(input_dim, config.num_experts)?;

        Ok(Self {
            layer_norm,
            projection,
            temperature: config.temperature,
            laplace_alpha: config.laplace_alpha,
            num_experts: config.num_experts,
            noise_std: config.noise_std,
        })
    }

    /// Get the number of experts.
    #[inline]
    #[must_use]
    pub fn num_experts(&self) -> usize {
        self.num_experts
    }

    /// Get the temperature value.
    #[inline]
    #[must_use]
    pub fn temperature(&self) -> f32 {
        self.temperature
    }

    /// Get the Laplace alpha value.
    #[inline]
    #[must_use]
    pub fn laplace_alpha(&self) -> f32 {
        self.laplace_alpha
    }

    /// Get the noise standard deviation.
    #[inline]
    #[must_use]
    pub fn noise_std(&self) -> f32 {
        self.noise_std
    }

    /// Set the temperature for softmax scaling.
    ///
    /// Lower temperature -> sharper distribution (more confident)
    /// Higher temperature -> flatter distribution (more uncertain)
    ///
    /// # Errors
    ///
    /// Returns `EmbeddingError::ConfigError` if temperature <= 0.
    pub fn set_temperature(&mut self, temperature: f32) -> EmbeddingResult<()> {
        if temperature <= 0.0 || temperature.is_nan() {
            return Err(EmbeddingError::ConfigError {
                message: format!("temperature must be > 0, got {}", temperature),
            });
        }
        self.temperature = temperature;
        Ok(())
    }

    /// Get a reference to the layer normalization component.
    #[inline]
    #[must_use]
    pub fn layer_norm(&self) -> &LayerNorm {
        &self.layer_norm
    }

    /// Get a reference to the linear projection component.
    #[inline]
    #[must_use]
    pub fn projection(&self) -> &Linear {
        &self.projection
    }
}
