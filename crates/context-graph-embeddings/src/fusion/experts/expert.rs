//! Single expert network implementation.
//!
//! This module implements the Expert struct which is a Feed-Forward Network
//! with a hidden layer used for embedding transformation.

use crate::error::{EmbeddingError, EmbeddingResult};
use crate::fusion::gating::Linear;
use tracing::debug;

use super::Activation;

/// Single expert network: Feed-Forward Network with hidden layer.
///
/// Architecture: input_dim -> hidden_dim -> activation -> output_dim
///
/// # Fields
///
/// - `input_to_hidden`: First linear layer (8320 -> 4096)
/// - `hidden_to_output`: Second linear layer (4096 -> 1536)
/// - `activation`: Activation function for hidden layer (GELU default)
/// - `expert_id`: Unique identifier (0..NUM_EXPERTS)
///
/// # Example
///
/// ```rust
/// use context_graph_embeddings::fusion::experts::{Expert, Activation};
/// use context_graph_embeddings::types::dimensions::{TOTAL_CONCATENATED, FUSED_OUTPUT};
///
/// let expert = Expert::new(0, TOTAL_CONCATENATED, 4096, FUSED_OUTPUT, Activation::Gelu).unwrap();
/// let input = vec![0.1f32; TOTAL_CONCATENATED];
/// let output = expert.forward(&input, 1).unwrap();
/// assert_eq!(output.len(), FUSED_OUTPUT);
/// ```
#[derive(Debug, Clone)]
pub struct Expert {
    /// First linear layer: input_dim -> hidden_dim
    input_to_hidden: Linear,
    /// Second linear layer: hidden_dim -> output_dim
    hidden_to_output: Linear,
    /// Activation function for hidden layer
    activation: Activation,
    /// Expert identifier (0-7)
    expert_id: usize,
    /// Input dimension
    input_dim: usize,
    /// Hidden dimension
    hidden_dim: usize,
    /// Output dimension
    output_dim: usize,
}

impl Expert {
    /// Create a new expert network.
    ///
    /// # Arguments
    ///
    /// * `expert_id` - Unique identifier 0..NUM_EXPERTS
    /// * `input_dim` - Input dimension (8320)
    /// * `hidden_dim` - Hidden layer dimension (4096)
    /// * `output_dim` - Output dimension (1536)
    /// * `activation` - Activation function (default: Gelu)
    ///
    /// # Errors
    ///
    /// Returns `EmbeddingError::InvalidDimension` if dimensions are invalid.
    ///
    /// # Example
    ///
    /// ```rust
    /// use context_graph_embeddings::fusion::experts::{Expert, Activation};
    ///
    /// let expert = Expert::new(0, 8320, 4096, 1536, Activation::Gelu).unwrap();
    /// assert_eq!(expert.expert_id(), 0);
    /// ```
    pub fn new(
        expert_id: usize,
        input_dim: usize,
        hidden_dim: usize,
        output_dim: usize,
        activation: Activation,
    ) -> EmbeddingResult<Self> {
        // Validate dimensions
        if input_dim == 0 {
            return Err(EmbeddingError::InvalidDimension {
                expected: 1,
                actual: 0,
            });
        }
        if hidden_dim == 0 {
            return Err(EmbeddingError::InvalidDimension {
                expected: 1,
                actual: 0,
            });
        }
        if output_dim == 0 {
            return Err(EmbeddingError::InvalidDimension {
                expected: 1,
                actual: 0,
            });
        }

        let input_to_hidden = Linear::new(input_dim, hidden_dim)?;
        let hidden_to_output = Linear::new(hidden_dim, output_dim)?;

        debug!(
            expert_id,
            input_dim, hidden_dim, output_dim, "Created Expert network"
        );

        Ok(Self {
            input_to_hidden,
            hidden_to_output,
            activation,
            expert_id,
            input_dim,
            hidden_dim,
            output_dim,
        })
    }

    /// Forward pass through single expert.
    ///
    /// # Arguments
    ///
    /// * `input` - Flattened input [batch_size * input_dim]
    /// * `batch_size` - Number of samples in batch
    ///
    /// # Returns
    ///
    /// * `Vec<f32>` of shape [batch_size * output_dim]
    ///
    /// # Errors
    ///
    /// * `EmbeddingError::EmptyInput` if batch_size is 0
    /// * `EmbeddingError::DimensionMismatch` if input length != batch_size * input_dim
    ///
    /// # Example
    ///
    /// ```rust
    /// use context_graph_embeddings::fusion::experts::{Expert, Activation};
    ///
    /// let expert = Expert::new(0, 100, 50, 25, Activation::Gelu).unwrap();
    /// let input = vec![0.1f32; 100];
    /// let output = expert.forward(&input, 1).unwrap();
    /// assert_eq!(output.len(), 25);
    /// ```
    pub fn forward(&self, input: &[f32], batch_size: usize) -> EmbeddingResult<Vec<f32>> {
        // Validate input
        if batch_size == 0 {
            return Err(EmbeddingError::EmptyInput);
        }

        let expected_len = batch_size * self.input_dim;
        if input.len() != expected_len {
            return Err(EmbeddingError::DimensionMismatch {
                expected: expected_len,
                got: input.len(),
            });
        }

        debug!(
            expert_id = self.expert_id,
            batch_size,
            input_len = input.len(),
            "Expert forward pass"
        );

        // Step 1: Input -> Hidden (linear)
        let hidden = self.input_to_hidden.forward(input, batch_size)?;

        // Step 2: Apply activation
        let activated: Vec<f32> = hidden.iter().map(|&x| self.activation.apply(x)).collect();

        // Step 3: Hidden -> Output (linear)
        self.hidden_to_output.forward(&activated, batch_size)
    }

    /// Get expert identifier.
    #[inline]
    #[must_use]
    pub fn expert_id(&self) -> usize {
        self.expert_id
    }

    /// Get input dimension.
    #[inline]
    #[must_use]
    pub fn input_dim(&self) -> usize {
        self.input_dim
    }

    /// Get hidden dimension.
    #[inline]
    #[must_use]
    pub fn hidden_dim(&self) -> usize {
        self.hidden_dim
    }

    /// Get output dimension.
    #[inline]
    #[must_use]
    pub fn output_dim(&self) -> usize {
        self.output_dim
    }

    /// Get activation function.
    #[inline]
    #[must_use]
    pub fn activation(&self) -> Activation {
        self.activation
    }

    /// Get parameter count for this expert.
    ///
    /// Calculated as:
    /// - input_to_hidden: input_dim * hidden_dim + hidden_dim (weights + bias)
    /// - hidden_to_output: hidden_dim * output_dim + output_dim (weights + bias)
    ///
    /// # Example
    ///
    /// For dimensions 8320 -> 4096 -> 1536:
    /// - Layer 1: 8320 * 4096 + 4096 = 34,082,816
    /// - Layer 2: 4096 * 1536 + 1536 = 6,293,088
    /// - Total: ~40.4M parameters per expert
    #[must_use]
    pub fn parameter_count(&self) -> usize {
        let layer1_params = self.input_dim * self.hidden_dim + self.hidden_dim;
        let layer2_params = self.hidden_dim * self.output_dim + self.output_dim;
        layer1_params + layer2_params
    }
}
