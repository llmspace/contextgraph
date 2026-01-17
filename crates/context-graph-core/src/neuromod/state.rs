//! NeuromodulationState - Central manager for all neuromodulators
//!
//! This module provides:
//! 1. `NeuromodulationState`: Complete state snapshot of all modulators
//! 2. `NeuromodulationManager`: Coordinates all neuromodulators
//! 3. `ModulatorType`: Enum for identifying modulators
//!
//! ## Architecture
//!
//! The manager owns DA, 5HT, and NE modulators. ACh is read-only, as it is
//! managed by the GWT MetaCognitiveLoop. The manager can:
//! - Get state snapshots for all modulators
//! - Apply homeostatic decay to all modulators
//! - Process trigger events
//! - Adjust individual modulators

use crate::error::{CoreError, CoreResult};
use serde::{Deserialize, Serialize};
use std::time::{Duration, Instant};

use super::dopamine::{DopamineLevel, DopamineModulator, DA_BASELINE, DA_MAX, DA_MIN};
use super::noradrenaline::{
    NoradrenalineLevel, NoradrenalineModulator, NE_BASELINE, NE_MAX, NE_MIN,
};
use super::serotonin::{
    SerotoninLevel, SerotoninModulator, NUM_EMBEDDING_SPACES, SEROTONIN_BASELINE, SEROTONIN_MAX,
    SEROTONIN_MIN,
};

/// Cascade configuration constants for cross-neuromodulator effects.
///
/// These constants define thresholds and deltas for cascade effects
/// between neuromodulators, implementing SPEC-NEURO-001 Section 8.2.
pub mod cascade {
    /// DA threshold for positive 5HT cascade (upper quartile of DA range [1,5])
    pub const DA_HIGH_THRESHOLD: f32 = 4.0;
    /// DA threshold for negative 5HT cascade (lower quartile of DA range [1,5])
    pub const DA_LOW_THRESHOLD: f32 = 2.0;
    /// 5HT adjustment magnitude for DA cascades (~5% of 5HT range [0,1])
    pub const SEROTONIN_CASCADE_DELTA: f32 = 0.05;
    /// DA change threshold for NE alertness cascade (~10% of DA range)
    pub const DA_CHANGE_THRESHOLD: f32 = 0.3;
    /// NE adjustment for significant DA change (~7% of NE range [0.5,2])
    pub const NE_ALERTNESS_DELTA: f32 = 0.1;
}

/// Report of cascade effects applied during goal progress.
///
/// Contains detailed information about all neuromodulator changes
/// triggered by `on_goal_progress_with_cascades()`.
#[derive(Debug, Clone, PartialEq)]
pub struct CascadeReport {
    /// DA delta actually applied (after clamping)
    pub da_delta: f32,
    /// New DA value after adjustment
    pub da_new: f32,
    /// 5HT delta applied from mood cascade (0.0 if not triggered)
    pub serotonin_delta: f32,
    /// New 5HT value after cascade
    pub serotonin_new: f32,
    /// NE delta applied from alertness cascade (0.0 if not triggered)
    pub ne_delta: f32,
    /// New NE value after cascade
    pub ne_new: f32,
    /// Whether mood cascade was triggered
    pub mood_cascade_triggered: bool,
    /// Whether alertness cascade was triggered
    pub alertness_cascade_triggered: bool,
}

impl Default for CascadeReport {
    fn default() -> Self {
        Self {
            da_delta: 0.0,
            da_new: 0.0,
            serotonin_delta: 0.0,
            serotonin_new: 0.0,
            ne_delta: 0.0,
            ne_new: 0.0,
            mood_cascade_triggered: false,
            alertness_cascade_triggered: false,
        }
    }
}

/// Complete neuromodulation state snapshot
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NeuromodulationState {
    /// Dopamine level and state
    pub dopamine: DopamineLevel,
    /// Serotonin level and state
    pub serotonin: SerotoninLevel,
    /// Noradrenaline level and state
    pub noradrenaline: NoradrenalineLevel,
    /// Acetylcholine level (read from GWT MetaCognitiveLoop)
    pub acetylcholine: f32,
    /// Timestamp of this state snapshot
    #[serde(with = "instant_serde")]
    pub timestamp: Instant,
}

// Custom serde for Instant (convert to/from duration since UNIX_EPOCH)
mod instant_serde {
    use serde::{Deserialize, Deserializer, Serialize, Serializer};
    use std::time::Instant;

    pub fn serialize<S>(instant: &Instant, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        // Serialize as elapsed duration in nanoseconds from some reference point
        let elapsed = instant.elapsed().as_nanos() as u64;
        elapsed.serialize(serializer)
    }

    pub fn deserialize<'de, D>(deserializer: D) -> Result<Instant, D::Error>
    where
        D: Deserializer<'de>,
    {
        // Deserialize as now - duration (approximation)
        let _nanos = u64::deserialize(deserializer)?;
        Ok(Instant::now())
    }
}

impl NeuromodulationState {
    /// Get Hopfield beta value (from dopamine)
    pub fn hopfield_beta(&self) -> f32 {
        self.dopamine.value
    }

    /// Get attention temperature (from noradrenaline)
    pub fn attention_temp(&self) -> f32 {
        self.noradrenaline.value
    }

    /// Get UTL learning rate (from acetylcholine)
    pub fn utl_learning_rate(&self) -> f32 {
        self.acetylcholine
    }

    /// Get serotonin-scaled weight for embedding space
    pub fn get_space_weight(&self, space_index: usize) -> f32 {
        if space_index >= NUM_EMBEDDING_SPACES {
            return 0.0;
        }
        let base_weight = self.serotonin.space_weights[space_index];
        let scaling = 0.5 + 0.5 * self.serotonin.value;
        base_weight * scaling
    }

    /// Check if system is in high alert state
    pub fn is_alert(&self) -> bool {
        self.noradrenaline.value > NE_BASELINE + 0.2
    }

    /// Check if system is in elevated learning state
    pub fn is_learning_elevated(&self) -> bool {
        self.acetylcholine > 0.001 + (0.002 - 0.001) / 2.0
    }
}

/// Modulator type enum for identification
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum ModulatorType {
    /// Dopamine - controls hopfield.beta
    Dopamine,
    /// Serotonin - controls space_weights
    Serotonin,
    /// Noradrenaline - controls attention.temp
    Noradrenaline,
    /// Acetylcholine - controls utl.lr (read-only, managed by GWT)
    Acetylcholine,
}

impl std::fmt::Display for ModulatorType {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            ModulatorType::Dopamine => write!(f, "Dopamine (DA)"),
            ModulatorType::Serotonin => write!(f, "Serotonin (5HT)"),
            ModulatorType::Noradrenaline => write!(f, "Noradrenaline (NE)"),
            ModulatorType::Acetylcholine => write!(f, "Acetylcholine (ACh)"),
        }
    }
}

/// Manager for all neuromodulators
#[derive(Debug)]
pub struct NeuromodulationManager {
    dopamine: DopamineModulator,
    serotonin: SerotoninModulator,
    noradrenaline: NoradrenalineModulator,
    last_update: Instant,
}

impl NeuromodulationManager {
    /// Create a new neuromodulation manager with all modulators at baseline
    pub fn new() -> Self {
        Self {
            dopamine: DopamineModulator::new(),
            serotonin: SerotoninModulator::new(),
            noradrenaline: NoradrenalineModulator::new(),
            last_update: Instant::now(),
        }
    }

    /// Get current state snapshot
    ///
    /// # Arguments
    /// - `ach_from_gwt`: Acetylcholine value read from GWT MetaCognitiveLoop
    pub fn get_state(&self, ach_from_gwt: f32) -> NeuromodulationState {
        NeuromodulationState {
            dopamine: self.dopamine.level().clone(),
            serotonin: self.serotonin.level().clone(),
            noradrenaline: self.noradrenaline.level().clone(),
            acetylcholine: ach_from_gwt,
            timestamp: Instant::now(),
        }
    }

    /// Apply homeostatic decay to all modulators
    pub fn decay_all(&mut self, delta_t: Duration) {
        self.dopamine.decay(delta_t);
        self.serotonin.decay(delta_t);
        self.noradrenaline.decay(delta_t);
        self.last_update = Instant::now();
    }

    /// Update with automatic delta_t calculation
    pub fn update(&mut self) {
        let delta_t = self.last_update.elapsed();
        self.decay_all(delta_t);
    }

    /// Adjust a specific modulator by delta
    ///
    /// # Arguments
    /// - `modulator`: Which modulator to adjust
    /// - `delta`: Amount to adjust (positive or negative)
    ///
    /// # Returns
    /// New value after adjustment
    ///
    /// # Errors
    /// - `CoreError::FeatureDisabled` if trying to adjust ACh (read-only)
    pub fn adjust(&mut self, modulator: ModulatorType, delta: f32) -> CoreResult<f32> {
        match modulator {
            ModulatorType::Dopamine => {
                let new_value = (self.dopamine.value() + delta).clamp(DA_MIN, DA_MAX);
                self.dopamine.set_value(new_value);
                Ok(new_value)
            }
            ModulatorType::Serotonin => {
                self.serotonin.adjust(delta);
                Ok(self.serotonin.value())
            }
            ModulatorType::Noradrenaline => {
                let new_value = (self.noradrenaline.value() + delta).clamp(NE_MIN, NE_MAX);
                self.noradrenaline.set_value(new_value);
                Ok(new_value)
            }
            ModulatorType::Acetylcholine => Err(CoreError::FeatureDisabled {
                feature:
                    "Direct ACh adjustment is disabled. ACh is managed by GWT MetaCognitiveLoop."
                        .to_string(),
            }),
        }
    }

    /// Set a specific modulator to exact value
    pub fn set(&mut self, modulator: ModulatorType, value: f32) -> CoreResult<f32> {
        match modulator {
            ModulatorType::Dopamine => {
                self.dopamine.set_value(value);
                Ok(self.dopamine.value())
            }
            ModulatorType::Serotonin => {
                self.serotonin.set_value(value);
                Ok(self.serotonin.value())
            }
            ModulatorType::Noradrenaline => {
                self.noradrenaline.set_value(value);
                Ok(self.noradrenaline.value())
            }
            ModulatorType::Acetylcholine => Err(CoreError::FeatureDisabled {
                feature: "Direct ACh setting is disabled. ACh is managed by GWT MetaCognitiveLoop."
                    .to_string(),
            }),
        }
    }

    /// Get current value of a modulator
    pub fn get(&self, modulator: ModulatorType) -> Option<f32> {
        match modulator {
            ModulatorType::Dopamine => Some(self.dopamine.value()),
            ModulatorType::Serotonin => Some(self.serotonin.value()),
            ModulatorType::Noradrenaline => Some(self.noradrenaline.value()),
            ModulatorType::Acetylcholine => None, // Must be read from GWT
        }
    }

    /// Get modulator range (min, baseline, max)
    pub fn get_range(modulator: ModulatorType) -> (f32, f32, f32) {
        match modulator {
            ModulatorType::Dopamine => (DA_MIN, DA_BASELINE, DA_MAX),
            ModulatorType::Serotonin => (SEROTONIN_MIN, SEROTONIN_BASELINE, SEROTONIN_MAX),
            ModulatorType::Noradrenaline => (NE_MIN, NE_BASELINE, NE_MAX),
            ModulatorType::Acetylcholine => (0.001, 0.001, 0.002),
        }
    }

    // Event handlers

    /// Handle workspace entry event (triggers dopamine)
    pub fn on_workspace_entry(&mut self) {
        self.dopamine.on_workspace_entry();
    }

    /// Handle threat detection event (triggers noradrenaline)
    pub fn on_threat_detected(&mut self) {
        self.noradrenaline.on_threat_detected();
    }

    /// Handle threat detection with severity
    pub fn on_threat_detected_with_severity(&mut self, severity: f32) {
        self.noradrenaline
            .on_threat_detected_with_severity(severity);
    }

    /// Handle safety confirmation (calms noradrenaline)
    pub fn on_safety_confirmed(&mut self) {
        self.noradrenaline.on_safety_confirmed();
    }

    /// Handle positive mood event (increases serotonin)
    pub fn on_positive_event(&mut self, magnitude: f32) {
        self.serotonin.on_positive_event(magnitude);
    }

    /// Handle negative mood event (decreases serotonin)
    pub fn on_negative_event(&mut self, magnitude: f32) {
        self.serotonin.on_negative_event(magnitude);
    }

    /// Handle goal progress from steering subsystem.
    ///
    /// Propagates goal achievement/regression to dopamine modulator.
    /// This provides direct neurochemical response to steering feedback.
    ///
    /// # Arguments
    /// * `delta` - Goal progress delta from SteeringReward.value [-1, 1]
    pub fn on_goal_progress(&mut self, delta: f32) {
        self.dopamine.on_goal_progress(delta);
    }

    /// Handle goal progress with cascade effects to other neuromodulators.
    ///
    /// Applies direct DA modulation, then triggers cascades to 5HT and NE.
    ///
    /// # Cascade Effects
    /// - DA > 4.0 after adjustment: 5HT += 0.05 (mood boost)
    /// - DA < 2.0 after adjustment: 5HT -= 0.05 (mood drop)
    /// - |DA_actual_change| > 0.3: NE += 0.1 (alertness spike)
    ///
    /// # Arguments
    /// * `delta` - Goal progress delta from steering [-1, 1]
    ///
    /// # Returns
    /// `CascadeReport` with all changes applied and new values
    pub fn on_goal_progress_with_cascades(&mut self, delta: f32) -> CascadeReport {
        // Guard against NaN - FAIL FAST
        if delta.is_nan() {
            tracing::warn!(
                "on_goal_progress_with_cascades received NaN delta - returning empty report"
            );
            return CascadeReport {
                da_new: self.dopamine.value(),
                serotonin_new: self.serotonin.value(),
                ne_new: self.noradrenaline.value(),
                ..Default::default()
            };
        }

        // Step 1: Capture DA before adjustment
        let da_old = self.dopamine.value();

        // Step 2: Apply direct DA modulation
        self.dopamine.on_goal_progress(delta);
        let da_new = self.dopamine.value();
        let da_actual_delta = da_new - da_old;

        // Step 3: Apply mood cascade (DA -> 5HT)
        let (serotonin_delta, mood_cascade_triggered) = self.apply_mood_cascade(da_new);
        let serotonin_new = self.serotonin.value();

        // Step 4: Apply alertness cascade (DA change -> NE)
        let (ne_delta, alertness_cascade_triggered) = self.apply_alertness_cascade(da_actual_delta);
        let ne_new = self.noradrenaline.value();

        // Step 5: Log cascade effects
        if mood_cascade_triggered || alertness_cascade_triggered {
            tracing::debug!(
                da_old = da_old,
                da_new = da_new,
                da_actual_delta = da_actual_delta,
                serotonin_delta = serotonin_delta,
                serotonin_new = serotonin_new,
                ne_delta = ne_delta,
                ne_new = ne_new,
                mood_cascade = mood_cascade_triggered,
                alertness_cascade = alertness_cascade_triggered,
                "Neuromodulation cascades applied"
            );
        }

        CascadeReport {
            da_delta: da_actual_delta,
            da_new,
            serotonin_delta,
            serotonin_new,
            ne_delta,
            ne_new,
            mood_cascade_triggered,
            alertness_cascade_triggered,
        }
    }

    /// Apply mood cascade: DA level affects 5HT
    /// Returns (serotonin_delta, triggered)
    fn apply_mood_cascade(&mut self, da_new: f32) -> (f32, bool) {
        if da_new > cascade::DA_HIGH_THRESHOLD {
            self.serotonin.adjust(cascade::SEROTONIN_CASCADE_DELTA);
            (cascade::SEROTONIN_CASCADE_DELTA, true)
        } else if da_new < cascade::DA_LOW_THRESHOLD {
            self.serotonin.adjust(-cascade::SEROTONIN_CASCADE_DELTA);
            (-cascade::SEROTONIN_CASCADE_DELTA, true)
        } else {
            (0.0, false)
        }
    }

    /// Apply alertness cascade: Significant DA change affects NE
    /// Returns (ne_delta, triggered)
    fn apply_alertness_cascade(&mut self, da_actual_delta: f32) -> (f32, bool) {
        if da_actual_delta.abs() > cascade::DA_CHANGE_THRESHOLD {
            let new_ne = self.noradrenaline.value() + cascade::NE_ALERTNESS_DELTA;
            self.noradrenaline.set_value(new_ne);
            (cascade::NE_ALERTNESS_DELTA, true)
        } else {
            (0.0, false)
        }
    }

    // Parameter accessors

    /// Get Hopfield beta (dopamine)
    pub fn get_hopfield_beta(&self) -> f32 {
        self.dopamine.get_hopfield_beta()
    }

    /// Get attention temperature (noradrenaline)
    pub fn get_attention_temp(&self) -> f32 {
        self.noradrenaline.get_attention_temp()
    }

    /// Get space weight for embedding index (serotonin)
    pub fn get_space_weight(&self, space_index: usize) -> f32 {
        self.serotonin.get_space_weight(space_index)
    }

    /// Get all space weights (serotonin)
    pub fn get_all_space_weights(&self) -> [f32; NUM_EMBEDDING_SPACES] {
        self.serotonin.get_all_space_weights()
    }

    /// Set base weight for embedding space
    pub fn set_space_weight(&mut self, space_index: usize, weight: f32) {
        self.serotonin.set_space_weight(space_index, weight);
    }

    // Alert states

    /// Check if system is alert (high NE)
    pub fn is_alert(&self) -> bool {
        self.noradrenaline.is_alert()
    }

    /// Check if system is in high alert (very high NE)
    pub fn is_high_alert(&self) -> bool {
        self.noradrenaline.is_high_alert()
    }

    /// Reset all modulators to baseline
    pub fn reset_all(&mut self) {
        self.dopamine.reset();
        self.serotonin.reset();
        self.noradrenaline.reset();
        self.last_update = Instant::now();
    }

    /// Get dopamine modulator reference
    pub fn dopamine(&self) -> &DopamineModulator {
        &self.dopamine
    }

    /// Get serotonin modulator reference
    pub fn serotonin(&self) -> &SerotoninModulator {
        &self.serotonin
    }

    /// Get noradrenaline modulator reference
    pub fn noradrenaline(&self) -> &NoradrenalineModulator {
        &self.noradrenaline
    }
}

impl Default for NeuromodulationManager {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_manager_creation() {
        let manager = NeuromodulationManager::new();

        assert!((manager.get_hopfield_beta() - DA_BASELINE).abs() < f32::EPSILON);
        assert!((manager.get_attention_temp() - NE_BASELINE).abs() < f32::EPSILON);
        assert!((manager.serotonin.value() - SEROTONIN_BASELINE).abs() < f32::EPSILON);
    }

    #[test]
    fn test_manager_get_state() {
        let manager = NeuromodulationManager::new();
        let state = manager.get_state(0.001);

        assert!((state.hopfield_beta() - DA_BASELINE).abs() < f32::EPSILON);
        assert!((state.attention_temp() - NE_BASELINE).abs() < f32::EPSILON);
        assert!((state.utl_learning_rate() - 0.001).abs() < f32::EPSILON);
    }

    #[test]
    fn test_manager_adjust_dopamine() {
        let mut manager = NeuromodulationManager::new();

        let result = manager.adjust(ModulatorType::Dopamine, 0.5);
        assert!(result.is_ok());
        assert!((result.unwrap() - (DA_BASELINE + 0.5)).abs() < f32::EPSILON);
    }

    #[test]
    fn test_manager_adjust_ach_fails() {
        let mut manager = NeuromodulationManager::new();

        let result = manager.adjust(ModulatorType::Acetylcholine, 0.0005);
        assert!(result.is_err());
    }

    #[test]
    fn test_manager_decay_all() {
        let mut manager = NeuromodulationManager::new();

        // Set all to max
        manager.dopamine.set_value(DA_MAX);
        manager.noradrenaline.set_value(NE_MAX);
        manager.serotonin.set_value(SEROTONIN_MAX);

        // Decay
        for _ in 0..50 {
            manager.decay_all(Duration::from_millis(100));
        }

        // All should be closer to baseline
        assert!(manager.dopamine.value() < DA_MAX);
        assert!(manager.noradrenaline.value() < NE_MAX);
        assert!(manager.serotonin.value() < SEROTONIN_MAX);
    }

    #[test]
    fn test_manager_workspace_entry() {
        let mut manager = NeuromodulationManager::new();
        let initial = manager.get_hopfield_beta();

        manager.on_workspace_entry();

        assert!(manager.get_hopfield_beta() > initial);
    }

    #[test]
    fn test_manager_threat_detection() {
        let mut manager = NeuromodulationManager::new();
        let initial = manager.get_attention_temp();

        manager.on_threat_detected();

        assert!(manager.get_attention_temp() > initial);
        assert!(manager.is_alert());
    }

    #[test]
    fn test_manager_get_range() {
        let (min, baseline, max) = NeuromodulationManager::get_range(ModulatorType::Dopamine);
        assert!((min - DA_MIN).abs() < f32::EPSILON);
        assert!((baseline - DA_BASELINE).abs() < f32::EPSILON);
        assert!((max - DA_MAX).abs() < f32::EPSILON);
    }

    #[test]
    fn test_manager_reset_all() {
        let mut manager = NeuromodulationManager::new();

        manager.on_workspace_entry();
        manager.on_threat_detected();
        manager.on_positive_event(1.0);

        manager.reset_all();

        assert!((manager.get_hopfield_beta() - DA_BASELINE).abs() < f32::EPSILON);
        assert!((manager.get_attention_temp() - NE_BASELINE).abs() < f32::EPSILON);
        assert!((manager.serotonin.value() - SEROTONIN_BASELINE).abs() < f32::EPSILON);
    }

    #[test]
    fn test_modulator_type_display() {
        assert_eq!(ModulatorType::Dopamine.to_string(), "Dopamine (DA)");
        assert_eq!(ModulatorType::Serotonin.to_string(), "Serotonin (5HT)");
        assert_eq!(
            ModulatorType::Noradrenaline.to_string(),
            "Noradrenaline (NE)"
        );
        assert_eq!(
            ModulatorType::Acetylcholine.to_string(),
            "Acetylcholine (ACh)"
        );
    }

    #[test]
    fn test_state_space_weights() {
        let manager = NeuromodulationManager::new();
        let state = manager.get_state(0.001);

        // Test all space weights are valid
        for i in 0..NUM_EMBEDDING_SPACES {
            let weight = state.get_space_weight(i);
            assert!((0.0..=1.0).contains(&weight));
        }

        // Invalid index should return 0
        assert!((state.get_space_weight(100) - 0.0).abs() < f32::EPSILON);
    }

    #[test]
    fn test_state_alert_states() {
        let mut manager = NeuromodulationManager::new();
        let state = manager.get_state(0.001);
        assert!(!state.is_alert());

        manager.on_threat_detected();
        let state = manager.get_state(0.001);
        assert!(state.is_alert());
    }

    #[test]
    fn test_state_learning_elevated() {
        let state_low = NeuromodulationState {
            dopamine: DopamineLevel::default(),
            serotonin: SerotoninLevel::default(),
            noradrenaline: NoradrenalineLevel::default(),
            acetylcholine: 0.001,
            timestamp: Instant::now(),
        };
        assert!(!state_low.is_learning_elevated());

        let state_high = NeuromodulationState {
            dopamine: DopamineLevel::default(),
            serotonin: SerotoninLevel::default(),
            noradrenaline: NoradrenalineLevel::default(),
            acetylcholine: 0.002,
            timestamp: Instant::now(),
        };
        assert!(state_high.is_learning_elevated());
    }

    // =========================================================================
    // on_goal_progress tests (TASK-NEURO-P2-001)
    // =========================================================================

    #[test]
    fn test_manager_on_goal_progress_positive() {
        use crate::neuromod::dopamine::DA_GOAL_SENSITIVITY;

        let mut manager = NeuromodulationManager::new();
        let initial = manager.get_hopfield_beta();

        manager.on_goal_progress(0.8);

        let expected = initial + 0.8 * DA_GOAL_SENSITIVITY;
        assert!(
            (manager.get_hopfield_beta() - expected).abs() < f32::EPSILON,
            "Expected {}, got {}",
            expected,
            manager.get_hopfield_beta()
        );
    }

    #[test]
    fn test_manager_on_goal_progress_negative() {
        use crate::neuromod::dopamine::DA_GOAL_SENSITIVITY;

        let mut manager = NeuromodulationManager::new();
        let initial = manager.get_hopfield_beta();

        manager.on_goal_progress(-0.6);

        let expected = initial - 0.6 * DA_GOAL_SENSITIVITY;
        assert!(
            (manager.get_hopfield_beta() - expected).abs() < f32::EPSILON,
            "Expected {}, got {}",
            expected,
            manager.get_hopfield_beta()
        );
    }

    // =========================================================================
    // Cascade Effect Tests (TASK-NEURO-P2-003)
    // =========================================================================

    #[test]
    fn test_cascade_high_da_boosts_serotonin() {
        use super::cascade;

        let mut manager = NeuromodulationManager::new();
        // Set DA just below high threshold
        manager.dopamine.set_value(3.95);
        let initial_5ht = manager.serotonin.value();

        // Large positive delta to push DA above 4.0
        // delta=1.0 * 0.1 sensitivity = 0.1 increase -> DA=4.05
        let report = manager.on_goal_progress_with_cascades(1.0);

        // Verify Source of Truth
        println!("=== HIGH DA -> 5HT CASCADE ===");
        println!("  DA before: 3.95, DA after: {}", manager.dopamine.value());
        println!(
            "  5HT before: {}, 5HT after: {}",
            initial_5ht,
            manager.serotonin.value()
        );
        println!("  Report: {:?}", report);

        assert!(
            report.da_new > cascade::DA_HIGH_THRESHOLD,
            "DA should exceed 4.0"
        );
        assert!(report.mood_cascade_triggered, "Mood cascade should trigger");
        assert!(
            (report.serotonin_delta - cascade::SEROTONIN_CASCADE_DELTA).abs() < f32::EPSILON,
            "5HT should increase by {}",
            cascade::SEROTONIN_CASCADE_DELTA
        );
        assert!(
            (manager.serotonin.value() - (initial_5ht + cascade::SEROTONIN_CASCADE_DELTA)).abs()
                < f32::EPSILON,
            "Source of Truth: 5HT actual value should be increased"
        );
    }

    #[test]
    fn test_cascade_low_da_lowers_serotonin() {
        use super::cascade;

        let mut manager = NeuromodulationManager::new();
        // Set DA just above low threshold
        manager.dopamine.set_value(2.05);
        let initial_5ht = manager.serotonin.value();

        // Large negative delta to push DA below 2.0
        // delta=-1.0 * 0.1 sensitivity = -0.1 decrease -> DA=1.95
        let report = manager.on_goal_progress_with_cascades(-1.0);

        println!("=== LOW DA -> 5HT CASCADE ===");
        println!("  DA before: 2.05, DA after: {}", manager.dopamine.value());
        println!(
            "  5HT before: {}, 5HT after: {}",
            initial_5ht,
            manager.serotonin.value()
        );

        assert!(
            report.da_new < cascade::DA_LOW_THRESHOLD,
            "DA should be below 2.0"
        );
        assert!(report.mood_cascade_triggered, "Mood cascade should trigger");
        assert!(
            (report.serotonin_delta + cascade::SEROTONIN_CASCADE_DELTA).abs() < f32::EPSILON,
            "5HT should decrease by {}",
            cascade::SEROTONIN_CASCADE_DELTA
        );
        assert!(
            (manager.serotonin.value() - (initial_5ht - cascade::SEROTONIN_CASCADE_DELTA)).abs()
                < f32::EPSILON,
            "Source of Truth: 5HT actual value should be decreased"
        );
    }

    #[test]
    fn test_cascade_significant_da_change_increases_ne() {
        use super::cascade;

        let mut manager = NeuromodulationManager::new();
        let initial_ne = manager.noradrenaline.value();

        // To get |DA_delta| > 0.3, we need input delta > 3.0 (since 3.0 * 0.1 = 0.3)
        // But delta is typically [-1, 1], so we need to test the helper directly

        // Test the helper directly by simulating large DA change
        let (ne_delta, triggered) = manager.apply_alertness_cascade(0.5); // Simulate large DA change

        println!("=== ALERTNESS CASCADE (direct) ===");
        println!("  Simulated DA change: 0.5");
        println!(
            "  NE before: {}, NE after: {}",
            initial_ne,
            manager.noradrenaline.value()
        );
        println!("  NE delta: {}, triggered: {}", ne_delta, triggered);

        assert!(
            triggered,
            "Alertness cascade should trigger for |delta|=0.5 > 0.3"
        );
        assert!(
            (ne_delta - cascade::NE_ALERTNESS_DELTA).abs() < f32::EPSILON,
            "NE should increase by {}",
            cascade::NE_ALERTNESS_DELTA
        );
        assert!(
            manager.noradrenaline.value() > initial_ne,
            "Source of Truth: NE actual value should be increased"
        );
    }

    #[test]
    fn test_cascade_no_trigger_in_normal_range() {
        let mut manager = NeuromodulationManager::new();
        // DA at baseline (3.0), small delta (0.1)
        let initial_5ht = manager.serotonin.value();
        let initial_ne = manager.noradrenaline.value();

        let report = manager.on_goal_progress_with_cascades(0.1);

        println!("=== NO CASCADE (normal range) ===");
        println!("  DA: {} -> {}", 3.0, report.da_new);
        println!("  5HT unchanged: {}", manager.serotonin.value());
        println!("  NE unchanged: {}", manager.noradrenaline.value());

        // DA change = 0.01 (below 0.3 threshold)
        // DA new = 3.01 (between 2.0 and 4.0)
        assert!(
            !report.mood_cascade_triggered,
            "Mood cascade should NOT trigger"
        );
        assert!(
            !report.alertness_cascade_triggered,
            "Alertness cascade should NOT trigger"
        );
        assert!(
            (manager.serotonin.value() - initial_5ht).abs() < f32::EPSILON,
            "5HT should be unchanged"
        );
        assert!(
            (manager.noradrenaline.value() - initial_ne).abs() < f32::EPSILON,
            "NE should be unchanged"
        );
    }

    #[test]
    fn test_cascade_report_accuracy() {
        use super::cascade;

        let mut manager = NeuromodulationManager::new();
        manager.dopamine.set_value(4.5); // High DA
        let _initial_5ht = manager.serotonin.value();

        let report = manager.on_goal_progress_with_cascades(0.5);

        println!("=== REPORT ACCURACY ===");
        println!(
            "  Report DA: new={}, delta={}",
            report.da_new, report.da_delta
        );
        println!(
            "  Report 5HT: delta={}, new={}",
            report.serotonin_delta, report.serotonin_new
        );
        println!("  Actual DA: {}", manager.dopamine.value());
        println!("  Actual 5HT: {}", manager.serotonin.value());

        // Verify report matches actual state
        assert!(
            (report.da_new - manager.dopamine.value()).abs() < f32::EPSILON,
            "Report DA should match actual"
        );
        assert!(
            (report.serotonin_new - manager.serotonin.value()).abs() < f32::EPSILON,
            "Report 5HT should match actual"
        );
        assert!(
            (report.ne_new - manager.noradrenaline.value()).abs() < f32::EPSILON,
            "Report NE should match actual"
        );

        // DA > 4.0, so mood cascade should trigger
        assert!(report.mood_cascade_triggered);
        assert!((report.serotonin_delta - cascade::SEROTONIN_CASCADE_DELTA).abs() < f32::EPSILON);
    }

    #[test]
    fn test_cascade_nan_handling() {
        let mut manager = NeuromodulationManager::new();
        let initial_da = manager.dopamine.value();
        let initial_5ht = manager.serotonin.value();
        let initial_ne = manager.noradrenaline.value();

        let report = manager.on_goal_progress_with_cascades(f32::NAN);

        println!("=== NaN HANDLING ===");
        println!("  DA unchanged: {}", manager.dopamine.value());
        println!("  5HT unchanged: {}", manager.serotonin.value());
        println!("  NE unchanged: {}", manager.noradrenaline.value());

        // Nothing should change
        assert!((manager.dopamine.value() - initial_da).abs() < f32::EPSILON);
        assert!((manager.serotonin.value() - initial_5ht).abs() < f32::EPSILON);
        assert!((manager.noradrenaline.value() - initial_ne).abs() < f32::EPSILON);
        assert!(!report.mood_cascade_triggered);
        assert!(!report.alertness_cascade_triggered);
    }

    #[test]
    fn test_cascade_serotonin_clamping() {
        let mut manager = NeuromodulationManager::new();

        // Test ceiling clamp: 5HT at max, high DA should not exceed max
        manager.serotonin.set_value(SEROTONIN_MAX);
        manager.dopamine.set_value(4.5);
        let _report = manager.on_goal_progress_with_cascades(0.1);

        println!("=== 5HT CEILING CLAMP ===");
        println!(
            "  5HT after cascade: {} (max={})",
            manager.serotonin.value(),
            SEROTONIN_MAX
        );

        assert!(
            manager.serotonin.value() <= SEROTONIN_MAX,
            "5HT must not exceed max"
        );

        // Test floor clamp: 5HT at min, low DA should not go below min
        manager.serotonin.set_value(SEROTONIN_MIN);
        manager.dopamine.set_value(1.5);
        let _report = manager.on_goal_progress_with_cascades(-0.1);

        println!("=== 5HT FLOOR CLAMP ===");
        println!(
            "  5HT after cascade: {} (min={})",
            manager.serotonin.value(),
            SEROTONIN_MIN
        );

        assert!(
            manager.serotonin.value() >= SEROTONIN_MIN,
            "5HT must not go below min"
        );
    }

    // =========================================================================
    // Full State Verification (FSV) tests (TASK-NEURO-P2-003)
    // =========================================================================

    #[test]
    fn test_fsv_cascade_source_of_truth() {
        use super::cascade;

        let mut manager = NeuromodulationManager::new();

        // === STEP 1: Establish baseline state ===
        manager.dopamine.set_value(3.95); // Just below high threshold

        println!("=== BEFORE STATE (Source of Truth) ===");
        println!("  DA value: {}", manager.dopamine.value());
        println!("  5HT value: {}", manager.serotonin.value());
        println!("  NE value: {}", manager.noradrenaline.value());

        let before_da = manager.dopamine.value();
        let before_5ht = manager.serotonin.value();
        let before_ne = manager.noradrenaline.value();

        // === STEP 2: Execute the cascade operation ===
        let report = manager.on_goal_progress_with_cascades(1.0);

        // === STEP 3: Read Source of Truth DIRECTLY ===
        println!("=== AFTER STATE (Source of Truth) ===");
        println!("  DA value: {}", manager.dopamine.value());
        println!("  5HT value: {}", manager.serotonin.value());
        println!("  NE value: {}", manager.noradrenaline.value());

        let after_da = manager.dopamine.value();
        let after_5ht = manager.serotonin.value();
        let after_ne = manager.noradrenaline.value();

        // === STEP 4: Verify changes match expectations ===
        println!("=== VERIFICATION ===");
        println!(
            "  DA change: {} -> {} (delta={})",
            before_da,
            after_da,
            after_da - before_da
        );
        println!(
            "  5HT change: {} -> {} (delta={})",
            before_5ht,
            after_5ht,
            after_5ht - before_5ht
        );
        println!(
            "  NE change: {} -> {} (delta={})",
            before_ne,
            after_ne,
            after_ne - before_ne
        );
        println!(
            "  Report matches actual: DA={}, 5HT={}, NE={}",
            (report.da_new - after_da).abs() < f32::EPSILON,
            (report.serotonin_new - after_5ht).abs() < f32::EPSILON,
            (report.ne_new - after_ne).abs() < f32::EPSILON
        );

        // DA should have increased
        assert!(after_da > before_da, "DA should increase");
        // DA > 4.0, so 5HT should increase
        assert!(
            after_da > cascade::DA_HIGH_THRESHOLD,
            "DA should exceed threshold"
        );
        assert!(
            (after_5ht - before_5ht - cascade::SEROTONIN_CASCADE_DELTA).abs() < f32::EPSILON,
            "5HT should increase by cascade delta"
        );
        // Report must match actual state
        assert!((report.da_new - after_da).abs() < f32::EPSILON);
        assert!((report.serotonin_new - after_5ht).abs() < f32::EPSILON);
    }

    #[test]
    fn test_edge_case_zero_delta() {
        let mut manager = NeuromodulationManager::new();
        let da_before = manager.dopamine.value();
        let sht_before = manager.serotonin.value();
        let ne_before = manager.noradrenaline.value();

        println!(
            "BEFORE: DA={}, 5HT={}, NE={}",
            manager.dopamine.value(),
            manager.serotonin.value(),
            manager.noradrenaline.value()
        );

        let report = manager.on_goal_progress_with_cascades(0.0);

        println!(
            "AFTER: DA={}, 5HT={}, NE={}",
            manager.dopamine.value(),
            manager.serotonin.value(),
            manager.noradrenaline.value()
        );

        // Verify: nothing changes with zero delta
        assert!(!report.mood_cascade_triggered);
        assert!(!report.alertness_cascade_triggered);
        assert!((report.da_delta).abs() < f32::EPSILON);
        assert!((manager.dopamine.value() - da_before).abs() < f32::EPSILON);
        assert!((manager.serotonin.value() - sht_before).abs() < f32::EPSILON);
        assert!((manager.noradrenaline.value() - ne_before).abs() < f32::EPSILON);
    }

    #[test]
    fn test_edge_case_da_at_ceiling() {
        let mut manager = NeuromodulationManager::new();
        manager.dopamine.set_value(DA_MAX);
        let initial_5ht = manager.serotonin.value();

        println!(
            "BEFORE: DA={} (max), 5HT={}",
            manager.dopamine.value(),
            initial_5ht
        );

        let report = manager.on_goal_progress_with_cascades(1.0);

        println!(
            "AFTER: DA={}, 5HT={}",
            manager.dopamine.value(),
            manager.serotonin.value()
        );

        // DA stays at max, 5HT still increases (DA > 4.0)
        assert!((manager.dopamine.value() - DA_MAX).abs() < f32::EPSILON);
        assert!(report.mood_cascade_triggered); // DA=5.0 > 4.0
        assert!(manager.serotonin.value() > initial_5ht);
    }

    #[test]
    fn test_edge_case_sequential_cascades() {
        let mut manager = NeuromodulationManager::new();
        manager.dopamine.set_value(3.95);

        println!("=== SEQUENTIAL CASCADE TEST ===");
        println!(
            "Initial: DA={}, 5HT={}",
            manager.dopamine.value(),
            manager.serotonin.value()
        );

        // First cascade
        let report1 = manager.on_goal_progress_with_cascades(1.0);
        println!(
            "After 1st: DA={}, 5HT={}, mood_triggered={}",
            manager.dopamine.value(),
            manager.serotonin.value(),
            report1.mood_cascade_triggered
        );

        // Second cascade
        let report2 = manager.on_goal_progress_with_cascades(1.0);
        println!(
            "After 2nd: DA={}, 5HT={}, mood_triggered={}",
            manager.dopamine.value(),
            manager.serotonin.value(),
            report2.mood_cascade_triggered
        );

        // Third cascade
        let report3 = manager.on_goal_progress_with_cascades(1.0);
        println!(
            "After 3rd: DA={}, 5HT={}, mood_triggered={}",
            manager.dopamine.value(),
            manager.serotonin.value(),
            report3.mood_cascade_triggered
        );

        // Each should trigger mood cascade if DA > 4.0
        // First call: DA=3.95+0.1=4.05 > 4.0 -> 5HT increases
        assert!(report1.mood_cascade_triggered);

        // 5HT should accumulate but clamp at max
        assert!(manager.serotonin.value() <= SEROTONIN_MAX);
        // DA should also clamp at max
        assert!(manager.dopamine.value() <= DA_MAX);
    }
}
