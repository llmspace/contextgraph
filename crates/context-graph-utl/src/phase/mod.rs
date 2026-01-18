//! Phase module.
//!
//! Implements consolidation phase detection (NREM/REM) and phase oscillation.
//! The PhaseOscillator provides a simplified phase angle for the UTL formula.

mod consolidation;
mod oscillator;

pub use consolidation::{ConsolidationPhase, PhaseDetector};
pub use oscillator::PhaseOscillator;
