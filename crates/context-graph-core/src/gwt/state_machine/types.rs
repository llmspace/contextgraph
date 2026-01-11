//! Consciousness State Machine Types
//!
//! Defines the state types for consciousness levels as specified in
//! Constitution v4.0.0 Section gwt.state_machine (lines 394-408).

use chrono::{DateTime, Utc};

/// Consciousness state
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ConsciousnessState {
    Dormant,
    Fragmented,
    Emerging,
    Conscious,
    Hypersync,
}

impl ConsciousnessState {
    /// Get human-readable name
    pub fn name(&self) -> &'static str {
        match self {
            Self::Dormant => "DORMANT",
            Self::Fragmented => "FRAGMENTED",
            Self::Emerging => "EMERGING",
            Self::Conscious => "CONSCIOUS",
            Self::Hypersync => "HYPERSYNC",
        }
    }

    /// Determine state from consciousness level
    pub fn from_level(level: f32) -> Self {
        match level {
            l if l > 0.95 => Self::Hypersync,
            l if l >= 0.8 => Self::Conscious,
            l if l >= 0.5 => Self::Emerging,
            l if l >= 0.3 => Self::Fragmented,
            _ => Self::Dormant,
        }
    }
}

/// State transition with timestamp and context
#[derive(Debug, Clone)]
pub struct StateTransition {
    pub from: ConsciousnessState,
    pub to: ConsciousnessState,
    pub timestamp: DateTime<Utc>,
    pub consciousness_level: f32,
}

/// Detailed transition analysis
#[derive(Debug, Clone)]
pub struct TransitionAnalysis {
    pub from_state: ConsciousnessState,
    pub to_state: ConsciousnessState,
    pub consciousness_delta: f32,
    pub coherence_was_increasing: bool,
    pub is_recovery: bool,
}
