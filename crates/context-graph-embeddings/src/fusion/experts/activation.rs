//! Activation functions for expert hidden layers.
//!
//! This module provides the activation function types used in expert networks.

/// Activation function types for expert hidden layers.
///
/// # Variants
///
/// - `Gelu`: Gaussian Error Linear Unit (default, recommended)
/// - `Relu`: Rectified Linear Unit (faster, less smooth)
/// - `Silu`: Sigmoid Linear Unit (smooth alternative)
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum Activation {
    /// Gaussian Error Linear Unit: x * 0.5 * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))
    #[default]
    Gelu,
    /// Rectified Linear Unit: max(0, x)
    Relu,
    /// Sigmoid Linear Unit: x * sigmoid(x)
    Silu,
}

impl Activation {
    /// Apply activation function element-wise.
    ///
    /// # Arguments
    ///
    /// * `x` - Input value
    ///
    /// # Returns
    ///
    /// Activated value according to the activation function.
    ///
    /// # Example
    ///
    /// ```rust
    /// use context_graph_embeddings::fusion::experts::Activation;
    ///
    /// let gelu = Activation::Gelu;
    /// let result = gelu.apply(1.0);
    /// assert!(result > 0.8 && result < 0.9); // GELU(1.0) ≈ 0.8413
    /// ```
    #[inline]
    pub fn apply(&self, x: f32) -> f32 {
        match self {
            Activation::Gelu => {
                // Approximation: x * 0.5 * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))
                const SQRT_2_OVER_PI: f32 = 0.797_884_6; // sqrt(2/pi)
                const COEF: f32 = 0.044715;
                let inner = SQRT_2_OVER_PI * (x + COEF * x * x * x);
                x * 0.5 * (1.0 + inner.tanh())
            }
            Activation::Relu => x.max(0.0),
            Activation::Silu => x * (1.0 / (1.0 + (-x).exp())), // x * sigmoid(x)
        }
    }
}
