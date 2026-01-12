//! Configuration for Poincare ball operations.
//!
//! Contains the [`PoincareBallConfig`] struct and its validation logic.

/// Configuration for Poincare ball operations.
///
/// Uses constitution-mandated values for numerical stability
/// during long random walks in REM phase.
#[derive(Debug, Clone, Copy)]
pub struct PoincareBallConfig {
    /// Maximum norm for valid points (< 1.0)
    /// Constitution: derived from boundary stability requirements
    pub max_norm: f32,

    /// Epsilon for numerical stability
    pub epsilon: f32,

    /// Curvature (negative for hyperbolic, standard = -1.0)
    pub curvature: f32,
}

impl Default for PoincareBallConfig {
    fn default() -> Self {
        Self {
            max_norm: 0.99999,
            epsilon: 1e-7,
            curvature: -1.0,
        }
    }
}

impl PoincareBallConfig {
    /// Validate configuration values.
    ///
    /// # Panics
    /// Panics with detailed error if configuration is invalid.
    pub fn validate(&self) {
        if self.max_norm >= 1.0 || self.max_norm <= 0.0 {
            panic!(
                "[POINCARE_WALK] Invalid max_norm at {}:{}: expected 0 < max_norm < 1.0, got {:.6}",
                file!(), line!(), self.max_norm
            );
        }
        if self.epsilon <= 0.0 || self.epsilon >= 1e-3 {
            panic!(
                "[POINCARE_WALK] Invalid epsilon at {}:{}: expected 0 < epsilon < 1e-3, got {:e}",
                file!(), line!(), self.epsilon
            );
        }
        if self.curvature >= 0.0 {
            panic!(
                "[POINCARE_WALK] Invalid curvature at {}:{}: expected < 0 for hyperbolic, got {:.6}",
                file!(), line!(), self.curvature
            );
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_config_default_values() {
        let config = PoincareBallConfig::default();
        assert!((config.max_norm - 0.99999).abs() < 1e-8);
        assert!((config.epsilon - 1e-7).abs() < 1e-10);
        assert!((config.curvature - (-1.0)).abs() < 1e-8);
    }

    #[test]
    fn test_config_validate_passes() {
        let config = PoincareBallConfig::default();
        config.validate(); // Should not panic
    }

    #[test]
    #[should_panic(expected = "[POINCARE_WALK] Invalid max_norm")]
    fn test_config_rejects_max_norm_too_high() {
        let config = PoincareBallConfig {
            max_norm: 1.0, // Invalid: must be < 1.0
            ..Default::default()
        };
        config.validate();
    }

    #[test]
    #[should_panic(expected = "[POINCARE_WALK] Invalid curvature")]
    fn test_config_rejects_positive_curvature() {
        let config = PoincareBallConfig {
            curvature: 1.0, // Invalid: must be < 0
            ..Default::default()
        };
        config.validate();
    }
}
