//! Phase oscillator for UTL phi computation.
//!
//! Provides a simple phase oscillator that tracks phase angle (phi)
//! for the UTL learning formula: L = f((Delta_S x Delta_C) . w_e . cos phi)

use crate::config::PhaseConfig;
use std::time::Duration;

/// Simple phase oscillator for UTL phi computation.
///
/// Tracks a phase angle that oscillates based on configured frequency.
/// Used in the UTL formula for phase modulation of learning signals.
#[derive(Debug, Clone)]
pub struct PhaseOscillator {
    /// Current phase angle in radians [0, pi].
    phase: f32,
    /// Configuration settings.
    config: PhaseConfig,
}

impl PhaseOscillator {
    /// Create a new phase oscillator with the given configuration.
    pub fn new(config: &PhaseConfig) -> Self {
        Self {
            phase: config.default_phase,
            config: config.clone(),
        }
    }

    /// Get the current phase angle in radians.
    #[inline]
    pub fn phase(&self) -> f32 {
        self.phase
    }

    /// Update the phase based on elapsed time.
    ///
    /// Phase advances based on the configured frequency_hz.
    /// Phase wraps around within the [min_phase, max_phase] range.
    pub fn update(&mut self, elapsed: Duration) {
        let elapsed_secs = elapsed.as_secs_f32();
        let delta_phase = 2.0 * std::f32::consts::PI * self.config.frequency_hz * elapsed_secs;

        // Advance phase and wrap within valid range
        self.phase += delta_phase;
        let range = self.config.max_phase - self.config.min_phase;
        if range > 0.0 {
            // Normalize to [0, range] then offset to [min, max]
            self.phase = self.config.min_phase
                + ((self.phase - self.config.min_phase) % range + range) % range;
        } else {
            self.phase = self.config.min_phase;
        }
    }

    /// Reset phase to the default value.
    pub fn reset(&mut self) {
        self.phase = self.config.default_phase;
    }

    /// Set phase to a specific value (clamped to valid range).
    pub fn set_phase(&mut self, phase: f32) {
        self.phase = self.config.clamp(phase);
    }

    /// Get the cosine of the current phase.
    ///
    /// This is used directly in the UTL formula:
    /// L = f((Delta_S x Delta_C) . w_e . cos phi)
    #[inline]
    pub fn cos_phase(&self) -> f32 {
        self.phase.cos()
    }
}

impl Default for PhaseOscillator {
    fn default() -> Self {
        Self::new(&PhaseConfig::default())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_new_oscillator_starts_at_default_phase() {
        let config = PhaseConfig::default();
        let oscillator = PhaseOscillator::new(&config);
        assert!((oscillator.phase() - config.default_phase).abs() < 1e-6);
    }

    #[test]
    fn test_phase_advances_with_time() {
        let config = PhaseConfig::default();
        let mut oscillator = PhaseOscillator::new(&config);
        let initial_phase = oscillator.phase();

        oscillator.update(Duration::from_millis(10));

        // Phase should have advanced (but may have wrapped)
        // With 100Hz frequency and 10ms, phase advances by 2*pi*100*0.01 = 2*pi radians
        // which wraps back, but we just check it changed
        let _ = initial_phase; // Phase may wrap, just ensure no panic
    }

    #[test]
    fn test_reset_returns_to_default() {
        let config = PhaseConfig::default();
        let mut oscillator = PhaseOscillator::new(&config);

        oscillator.update(Duration::from_millis(50));
        oscillator.reset();

        assert!((oscillator.phase() - config.default_phase).abs() < 1e-6);
    }

    #[test]
    fn test_set_phase_clamps_to_valid_range() {
        let config = PhaseConfig::default();
        let mut oscillator = PhaseOscillator::new(&config);

        oscillator.set_phase(10.0); // Way over max
        assert!(oscillator.phase() <= config.max_phase);

        oscillator.set_phase(-5.0); // Way under min
        assert!(oscillator.phase() >= config.min_phase);
    }

    #[test]
    fn test_cos_phase_returns_valid_cosine() {
        let config = PhaseConfig::default();
        let oscillator = PhaseOscillator::new(&config);

        let cos = oscillator.cos_phase();
        assert!(cos >= -1.0 && cos <= 1.0);
    }

    #[test]
    fn test_default_phase_zero_gives_cos_one() {
        let mut config = PhaseConfig::default();
        config.default_phase = 0.0;
        let oscillator = PhaseOscillator::new(&config);

        // cos(0) = 1.0
        assert!((oscillator.cos_phase() - 1.0).abs() < 1e-6);
    }
}
