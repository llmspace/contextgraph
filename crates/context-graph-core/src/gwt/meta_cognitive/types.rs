//! Meta-Cognitive Types and Constants
//!
//! Type definitions for the meta-cognitive feedback loop system.

use chrono::{DateTime, Utc};
use std::collections::VecDeque;

/// Acetylcholine baseline level (minimum learning rate)
/// Constitution v4.0.0: neuromod.Acetylcholine.range = "[0.001, 0.002]"
pub const ACH_BASELINE: f32 = 0.001;

/// Acetylcholine maximum level
pub const ACH_MAX: f32 = 0.002;

/// Acetylcholine decay rate per evaluation (homeostatic regulation)
/// Decays toward baseline when dream is not triggered
pub const ACH_DECAY_RATE: f32 = 0.1;

/// Meta-cognitive learning loop state
#[derive(Debug, Clone)]
pub struct MetaCognitiveLoop {
    /// Recent meta-scores (for trend detection)
    pub(crate) recent_scores: VecDeque<f32>,
    /// Maximum history to keep
    pub(crate) max_history: usize,
    /// Count of consecutive low scores
    pub(crate) consecutive_low_scores: u32,
    /// Count of consecutive high scores
    pub(crate) consecutive_high_scores: u32,
    /// Current Acetylcholine level (learning rate modulator)
    pub(crate) acetylcholine_level: f32,
    /// Current monitoring frequency (samples per second)
    pub(crate) monitoring_frequency: f32,
    /// Last time meta-score was calculated
    pub(crate) last_update: DateTime<Utc>,
}

/// Result of a meta-cognitive evaluation
#[derive(Debug, Clone)]
pub struct MetaCognitiveState {
    /// Current meta-score
    pub meta_score: f32,
    /// Average meta-score over recent history
    pub avg_meta_score: f32,
    /// Trend (increasing/decreasing/stable)
    pub trend: ScoreTrend,
    /// Current Acetylcholine level
    pub acetylcholine: f32,
    /// Whether introspective dream is triggered
    pub dream_triggered: bool,
    /// Whether monitoring frequency adjustment is needed
    pub frequency_adjustment: FrequencyAdjustment,
}

/// Score trend direction
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ScoreTrend {
    Increasing,
    Decreasing,
    Stable,
}

/// Frequency adjustment recommendation
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FrequencyAdjustment {
    None,
    Increase, // High confidence - monitor less frequently
    Decrease, // Low confidence - monitor more frequently
}

/// Neuromodulation effects from meta-cognitive adjustments
#[derive(Debug, Clone)]
pub struct NeuromodulationEffect {
    /// Acetylcholine adjustment (learning rate)
    pub acetylcholine_delta: f32,
    /// Whether to trigger introspective dream
    pub trigger_introspective_dream: bool,
    /// Suggested monitoring interval (milliseconds)
    pub monitoring_interval_ms: u32,
}
