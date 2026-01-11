//! Core Handlers struct and dispatch logic.
//!
//! TASK-S001: Updated to use TeleologicalMemoryStore and MultiArrayEmbeddingProvider.
//! TASK-S003: Added GoalAlignmentCalculator for purpose/goal operations.
//! TASK-S004: Added JohariTransitionManager for johari/* handlers.
//! TASK-S005: Added MetaUtlTracker for meta_utl/* handlers.
//! TASK-GWT-001: Added GWT/Kuramoto provider traits for consciousness operations.
//! NO BACKWARDS COMPATIBILITY with legacy MemoryStore trait.

use std::collections::HashMap;
use std::sync::Arc;
use std::time::Instant;

use parking_lot::RwLock;
use tracing::debug;
use uuid::Uuid;

use context_graph_core::alignment::GoalAlignmentCalculator;
use context_graph_core::atc::AdaptiveThresholdCalibration;
use context_graph_core::dream::{AmortizedLearner, DreamController, DreamScheduler};
use context_graph_core::johari::{DynDefaultJohariManager, JohariTransitionManager, NUM_EMBEDDERS};
use context_graph_core::monitoring::{
    LayerStatusProvider, StubLayerStatusProvider, StubSystemMonitor, SystemMonitor,
};
use context_graph_core::neuromod::NeuromodulationManager;
use context_graph_core::purpose::GoalHierarchy;
use context_graph_core::traits::{
    MultiArrayEmbeddingProvider, TeleologicalMemoryStore, UtlProcessor,
};

// TASK-GWT-001: Import GWT provider traits
use super::gwt_traits::{
    GwtSystemProvider, KuramotoProvider, MetaCognitiveProvider, SelfEgoProvider, WorkspaceProvider,
};

/// Prediction type for tracking
/// TASK-S005: Used to distinguish storage vs retrieval predictions.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum PredictionType {
    Storage,
    Retrieval,
}

/// Domain enum for domain-specific accuracy tracking.
/// TASK-METAUTL-P0-001: Enables per-domain meta-learning optimization.
#[allow(dead_code)]
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum Domain {
    /// Source code, programming-related content
    Code,
    /// Medical and healthcare content
    Medical,
    /// Legal documents and regulations
    Legal,
    /// Creative writing, art, design
    Creative,
    /// Research papers, scientific content
    Research,
    /// General purpose, unclassified
    General,
}

impl Default for Domain {
    fn default() -> Self {
        Self::General
    }
}

/// Meta-learning event types for logging and auditing.
/// TASK-METAUTL-P0-001: Used to track significant meta-learning state changes.
#[allow(dead_code)]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum MetaLearningEventType {
    /// Lambda weight adjustment occurred
    LambdaAdjustment,
    /// Bayesian optimization escalation triggered
    BayesianEscalation,
    /// Accuracy dropped below threshold
    AccuracyAlert,
    /// Recovery from low accuracy period
    AccuracyRecovery,
    /// Weight clamping applied (exceeded bounds)
    WeightClamped,
}

/// Meta-learning event for logging significant state changes.
/// TASK-METAUTL-P0-001: Provides audit trail for meta-learning behavior.
#[allow(dead_code)]
#[derive(Clone, Debug)]
pub struct MetaLearningEvent {
    /// Type of event
    pub event_type: MetaLearningEventType,
    /// When the event occurred
    pub timestamp: Instant,
    /// Embedder index affected (if applicable)
    pub embedder_index: Option<usize>,
    /// Previous value (if applicable)
    pub previous_value: Option<f32>,
    /// New value (if applicable)
    pub new_value: Option<f32>,
    /// Optional description
    pub description: Option<String>,
}

#[allow(dead_code)]
impl MetaLearningEvent {
    /// Create a lambda adjustment event.
    pub fn lambda_adjustment(embedder_idx: usize, previous: f32, new: f32) -> Self {
        Self {
            event_type: MetaLearningEventType::LambdaAdjustment,
            timestamp: Instant::now(),
            embedder_index: Some(embedder_idx),
            previous_value: Some(previous),
            new_value: Some(new),
            description: None,
        }
    }

    /// Create a bayesian escalation event.
    pub fn bayesian_escalation(consecutive_low: usize) -> Self {
        Self {
            event_type: MetaLearningEventType::BayesianEscalation,
            timestamp: Instant::now(),
            embedder_index: None,
            previous_value: None,
            new_value: Some(consecutive_low as f32),
            description: Some(format!(
                "Escalation triggered after {} consecutive low accuracy cycles",
                consecutive_low
            )),
        }
    }

    /// Create a weight clamped event.
    pub fn weight_clamped(embedder_idx: usize, original: f32, clamped: f32) -> Self {
        Self {
            event_type: MetaLearningEventType::WeightClamped,
            timestamp: Instant::now(),
            embedder_index: Some(embedder_idx),
            previous_value: Some(original),
            new_value: Some(clamped),
            description: None,
        }
    }
}

/// Configuration for self-correction behavior.
/// TASK-METAUTL-P0-001: Constitution-defined parameters for meta-learning.
#[allow(dead_code)]
#[derive(Clone, Debug)]
pub struct SelfCorrectionConfig {
    /// Whether self-correction is enabled
    pub enabled: bool,
    /// Prediction error threshold (constitution: 0.2)
    pub error_threshold: f32,
    /// Maximum consecutive failures before escalation (constitution: 10)
    pub max_consecutive_failures: usize,
    /// Accuracy threshold below which is considered "low" (constitution: 0.7)
    pub low_accuracy_threshold: f32,
    /// Minimum weight bound (constitution NORTH-016: 0.05)
    /// Note: 13 × 0.05 = 0.65 < 1.0, so sum=1.0 is achievable
    pub min_weight: f32,
    /// Maximum weight bound (constitution: 0.9)
    pub max_weight: f32,
    /// Escalation strategy
    pub escalation_strategy: String,
}

impl Default for SelfCorrectionConfig {
    /// Creates config with constitution-mandated defaults.
    ///
    /// From docs2/constitution.yaml:
    /// - threshold: 0.2
    /// - max_consecutive_failures: 10
    /// - escalation_strategy: "bayesian_optimization"
    /// - NORTH-016_WeightAdjuster: min=0.05, max_delta=0.10
    fn default() -> Self {
        Self {
            enabled: true,
            error_threshold: 0.2,
            max_consecutive_failures: 10,
            low_accuracy_threshold: 0.7,
            min_weight: 0.05, // NORTH-016: min=0.05 (13×0.05=0.65 < 1.0, sum is achievable)
            max_weight: 0.9,
            escalation_strategy: "bayesian_optimization".to_string(),
        }
    }
}

/// Stored prediction for validation
/// TASK-S005: Stores predicted values for later validation against actual outcomes.
#[derive(Clone, Debug)]
pub struct StoredPrediction {
    pub _created_at: Instant,
    pub prediction_type: PredictionType,
    pub predicted_values: serde_json::Value,
    #[allow(dead_code)]
    pub fingerprint_id: Uuid,
}

/// Meta-UTL Tracker for learning about learning
///
/// TASK-S005: Tracks per-embedder accuracy, pending predictions, and optimized weights.
/// TASK-METAUTL-P0-001: Extended with consecutive low tracking and weight clamping.
/// Uses rolling window for accuracy tracking to maintain recency bias.
#[derive(Debug)]
pub struct MetaUtlTracker {
    /// Pending predictions awaiting validation
    pub pending_predictions: HashMap<Uuid, StoredPrediction>,
    /// Per-embedder accuracy rolling window (100 samples per embedder)
    pub embedder_accuracy: [[f32; 100]; NUM_EMBEDDERS],
    /// Current index in each embedder's rolling window
    pub accuracy_indices: [usize; NUM_EMBEDDERS],
    /// Number of samples in each embedder's rolling window
    pub accuracy_counts: [usize; NUM_EMBEDDERS],
    /// Current optimized weights (sum to 1.0, clamped to [0.05, 0.9] per constitution)
    pub current_weights: [f32; NUM_EMBEDDERS],
    /// Total predictions made
    pub prediction_count: usize,
    /// Total validations completed
    pub validation_count: usize,
    /// Last weight update timestamp
    pub last_weight_update: Option<Instant>,
    /// TASK-METAUTL-P0-001: Consecutive cycles with accuracy < 0.7
    pub consecutive_low_count: usize,
    /// TASK-METAUTL-P0-001: Whether Bayesian escalation has been triggered
    pub escalation_triggered: bool,
    /// TASK-METAUTL-P0-001: Self-correction configuration
    pub config: SelfCorrectionConfig,
    /// TASK-METAUTL-P0-001: Tracks which embedders have been updated in current cycle
    cycle_embedder_updated: [bool; NUM_EMBEDDERS],
    /// TASK-METAUTL-P0-001: Number of complete accuracy recording cycles
    cycle_count: usize,
}

impl Default for MetaUtlTracker {
    fn default() -> Self {
        // Initialize with uniform weights (1/13 each)
        let initial_weight = 1.0 / NUM_EMBEDDERS as f32;
        Self {
            pending_predictions: HashMap::new(),
            embedder_accuracy: [[0.0; 100]; NUM_EMBEDDERS],
            accuracy_indices: [0; NUM_EMBEDDERS],
            accuracy_counts: [0; NUM_EMBEDDERS],
            current_weights: [initial_weight; NUM_EMBEDDERS],
            prediction_count: 0,
            validation_count: 0,
            last_weight_update: None,
            // TASK-METAUTL-P0-001: Initialize consecutive tracking
            consecutive_low_count: 0,
            escalation_triggered: false,
            config: SelfCorrectionConfig::default(),
            cycle_embedder_updated: [false; NUM_EMBEDDERS],
            cycle_count: 0,
        }
    }
}

impl MetaUtlTracker {
    /// Weight precision tolerance for sum normalization
    #[allow(dead_code)]
    const WEIGHT_PRECISION: f32 = 1e-6;

    /// Threshold for detecting accuracy trend changes
    #[allow(dead_code)]
    const TREND_THRESHOLD: f32 = 0.02;

    /// Create a new MetaUtlTracker
    pub fn new() -> Self {
        Self::default()
    }

    /// Store a prediction for later validation
    pub fn store_prediction(&mut self, prediction_id: Uuid, prediction: StoredPrediction) {
        self.pending_predictions.insert(prediction_id, prediction);
        self.prediction_count += 1;
    }

    /// Get a pending prediction by ID
    #[allow(dead_code)]
    pub fn get_prediction(&self, prediction_id: &Uuid) -> Option<&StoredPrediction> {
        self.pending_predictions.get(prediction_id)
    }

    /// Remove and return a prediction (for validation)
    pub fn remove_prediction(&mut self, prediction_id: &Uuid) -> Option<StoredPrediction> {
        self.pending_predictions.remove(prediction_id)
    }

    /// Record accuracy for an embedder
    ///
    /// TASK-METAUTL-P0-001: Also tracks consecutive low accuracy cycles.
    /// A "cycle" is considered complete when all 13 embedders have been updated.
    /// Low accuracy is defined as < 0.7 (constitution).
    pub fn record_accuracy(&mut self, embedder_index: usize, accuracy: f32) {
        if embedder_index >= NUM_EMBEDDERS {
            debug!(
                embedder_index = embedder_index,
                "record_accuracy: invalid embedder index, ignoring"
            );
            return;
        }

        // Clamp accuracy to [0.0, 1.0]
        let clamped_accuracy = accuracy.clamp(0.0, 1.0);

        let idx = self.accuracy_indices[embedder_index];
        self.embedder_accuracy[embedder_index][idx] = clamped_accuracy;
        self.accuracy_indices[embedder_index] = (idx + 1) % 100;
        if self.accuracy_counts[embedder_index] < 100 {
            self.accuracy_counts[embedder_index] += 1;
        }

        // TASK-METAUTL-P0-001: Track cycle completion
        self.cycle_embedder_updated[embedder_index] = true;

        // Check if a complete cycle has occurred (all embedders updated)
        if self.cycle_embedder_updated.iter().all(|&updated| updated) {
            self.cycle_count += 1;
            // Reset cycle tracking for next cycle
            self.cycle_embedder_updated = [false; NUM_EMBEDDERS];
            // Check consecutive low accuracy at end of cycle
            self.check_consecutive_low_accuracy();
        }
    }

    /// Check if overall accuracy is low and track consecutive count.
    ///
    /// TASK-METAUTL-P0-001: Called at the END of each complete cycle (when all
    /// 13 embedders have been recorded). This ensures we count cycles, not
    /// individual record_accuracy calls.
    fn check_consecutive_low_accuracy(&mut self) {
        // Calculate overall accuracy across all embedders
        let mut total_accuracy = 0.0f32;
        let mut embedder_count = 0usize;

        for i in 0..NUM_EMBEDDERS {
            if let Some(acc) = self.get_embedder_accuracy(i) {
                total_accuracy += acc;
                embedder_count += 1;
            }
        }

        // Only check if we have data from all embedders
        if embedder_count < NUM_EMBEDDERS {
            return;
        }

        let overall_accuracy = total_accuracy / embedder_count as f32;
        let threshold = self.config.low_accuracy_threshold; // 0.7

        if overall_accuracy < threshold {
            self.consecutive_low_count += 1;
            debug!(
                overall_accuracy = overall_accuracy,
                threshold = threshold,
                consecutive_low_count = self.consecutive_low_count,
                cycle_count = self.cycle_count,
                "Meta-UTL: low accuracy cycle recorded"
            );

            // Check if escalation should be triggered
            if self.consecutive_low_count >= self.config.max_consecutive_failures
                && !self.escalation_triggered
            {
                self.escalation_triggered = true;
                tracing::warn!(
                    consecutive_low = self.consecutive_low_count,
                    threshold = self.config.max_consecutive_failures,
                    "TASK-METAUTL-P0-001: Bayesian escalation triggered"
                );
            }
        } else {
            // Reset consecutive count on recovery
            if self.consecutive_low_count > 0 {
                debug!(
                    previous_count = self.consecutive_low_count,
                    overall_accuracy = overall_accuracy,
                    "Meta-UTL: accuracy recovered, resetting consecutive low count"
                );
            }
            self.consecutive_low_count = 0;
            // Note: We don't reset escalation_triggered here - that requires explicit reset
        }
    }

    /// Get average accuracy for an embedder
    pub fn get_embedder_accuracy(&self, embedder_index: usize) -> Option<f32> {
        if embedder_index >= NUM_EMBEDDERS || self.accuracy_counts[embedder_index] == 0 {
            return None;
        }
        let count = self.accuracy_counts[embedder_index];
        let sum: f32 = self.embedder_accuracy[embedder_index][..count].iter().sum();
        Some(sum / count as f32)
    }

    /// Get accuracy trend for an embedder (recent vs older samples)
    pub fn get_accuracy_trend(&self, embedder_index: usize) -> Option<&'static str> {
        if embedder_index >= NUM_EMBEDDERS || self.accuracy_counts[embedder_index] < 10 {
            return None;
        }
        let count = self.accuracy_counts[embedder_index];
        let recent_start = count.saturating_sub(10);
        let recent_sum: f32 = self.embedder_accuracy[embedder_index][recent_start..count]
            .iter()
            .sum();
        let recent_avg = recent_sum / 10.0;

        let older_end = if count >= 20 {
            count - 10
        } else {
            count - (count / 2)
        };
        let older_start = older_end.saturating_sub(10);
        let older_sum: f32 = self.embedder_accuracy[embedder_index][older_start..older_end]
            .iter()
            .sum();
        let older_count = older_end - older_start;
        if older_count == 0 {
            return Some("stable");
        }
        let older_avg = older_sum / older_count as f32;

        if recent_avg > older_avg + Self::TREND_THRESHOLD {
            Some("improving")
        } else if recent_avg < older_avg - Self::TREND_THRESHOLD {
            Some("declining")
        } else {
            Some("stable")
        }
    }

    /// Redistribute surplus from over-max weights to below-max weights.
    /// Returns the total surplus that was redistributed.
    fn redistribute_excess_weight(&mut self, max_weight: f32) -> f32 {
        let mut total_surplus = 0.0f32;
        let mut capped_count = 0usize;

        // Find weights above max
        for &weight in self.current_weights.iter() {
            if weight > max_weight {
                total_surplus += weight - max_weight;
                capped_count += 1;
            }
        }

        if total_surplus < Self::WEIGHT_PRECISION {
            return 0.0; // No surplus to redistribute
        }

        // Count weights below max that can receive redistribution
        let below_max_count = NUM_EMBEDDERS - capped_count;
        if below_max_count == 0 {
            // All weights at or above max - just cap them all
            for weight in self.current_weights.iter_mut() {
                if *weight > max_weight {
                    debug!(
                        original_weight = *weight,
                        clamped_weight = max_weight,
                        "TASK-METAUTL-P0-001: Lambda weight clamped to maximum"
                    );
                    *weight = max_weight;
                }
            }
            return total_surplus;
        }

        // Calculate how much each below-max weight should receive
        let redistribution = total_surplus / below_max_count as f32;

        // Apply capping and redistribution
        for weight in self.current_weights.iter_mut() {
            if *weight > max_weight {
                debug!(
                    original_weight = *weight,
                    clamped_weight = max_weight,
                    "TASK-METAUTL-P0-001: Lambda weight clamped to maximum"
                );
                *weight = max_weight;
            } else {
                *weight += redistribution;
            }
        }

        total_surplus
    }

    /// Update weights based on accuracy (called every 100 validations)
    ///
    /// TASK-METAUTL-P0-001: REQ-METAUTL-006/007 compliance.
    ///
    /// Priority of constraints:
    /// 1. Sum = 1.0 (REQ-METAUTL-006, HARD)
    /// 2. Max weight ≤ 0.9 (HARD - prevents single embedder dominance)
    /// 3. Min weight ≥ 0.05 (SOFT - best effort, may be violated in extreme cases)
    ///
    /// Algorithm:
    /// 1. Normalize weights based on accuracy (sum=1.0)
    /// 2. Cap any weight above max, redistribute surplus proportionally
    /// 3. Final normalization to ensure exact sum=1.0
    pub fn update_weights(&mut self) {
        // Calculate average accuracy per embedder
        let mut accuracies = [0.0f32; NUM_EMBEDDERS];
        let mut total_accuracy = 0.0f32;

        for (i, acc) in accuracies.iter_mut().enumerate() {
            *acc = self
                .get_embedder_accuracy(i)
                .unwrap_or(1.0 / NUM_EMBEDDERS as f32);
            total_accuracy += *acc;
        }

        // Normalize to get initial weights (sum = 1.0)
        if total_accuracy > 0.0 {
            for (weight, &acc) in self.current_weights.iter_mut().zip(accuracies.iter()) {
                *weight = acc / total_accuracy;
            }
        }

        let max_weight = self.config.max_weight; // 0.9
        let mut clamping_occurred = false;

        // STEP 1: Cap weights above max and redistribute surplus
        // This is the HARD constraint for max weight
        // Loop until no more surplus needs redistribution
        loop {
            let surplus = self.redistribute_excess_weight(max_weight);
            if surplus > 0.0 {
                clamping_occurred = true;
            }
            if surplus < Self::WEIGHT_PRECISION {
                break;
            }
        }

        // STEP 2: Final normalization to ensure exact sum=1.0
        let weight_sum: f32 = self.current_weights.iter().sum();
        if (weight_sum - 1.0).abs() > Self::WEIGHT_PRECISION {
            let scale = 1.0 / weight_sum;
            for weight in self.current_weights.iter_mut() {
                *weight *= scale;
            }
        }

        // Note: min_weight is a SOFT constraint. In extreme distributions
        // where one embedder dominates, other weights may be below min_weight
        // to maintain sum=1.0. This is mathematically necessary.
        // See EC-001 test for documentation.

        self.last_weight_update = Some(Instant::now());

        tracing::info!(
            validation_count = self.validation_count,
            weights_sum = self.current_weights.iter().sum::<f32>(),
            clamping_occurred = clamping_occurred,
            "Meta-UTL weights updated"
        );
    }

    /// Increment validation count and check if weights need update
    pub fn record_validation(&mut self) {
        self.validation_count += 1;
        if self.validation_count.is_multiple_of(100) {
            self.update_weights();
        }
    }

    /// Check if Bayesian escalation is needed.
    ///
    /// TASK-METAUTL-P0-001: Returns true when accuracy has been below 0.7
    /// for 10 or more consecutive cycles.
    pub fn needs_escalation(&self) -> bool {
        self.escalation_triggered
    }

    /// Get the current consecutive low accuracy count.
    ///
    /// TASK-METAUTL-P0-001: Returns the number of consecutive cycles
    /// with overall accuracy below 0.7.
    pub fn consecutive_low_count(&self) -> usize {
        self.consecutive_low_count
    }

    /// Reset the consecutive low accuracy counter and escalation flag.
    ///
    /// TASK-METAUTL-P0-001: Call this after taking corrective action
    /// (e.g., after Bayesian optimization completes).
    pub fn reset_consecutive_low(&mut self) {
        if self.consecutive_low_count > 0 || self.escalation_triggered {
            debug!(
                previous_count = self.consecutive_low_count,
                was_escalated = self.escalation_triggered,
                "TASK-METAUTL-P0-001: Resetting consecutive low tracking"
            );
        }
        self.consecutive_low_count = 0;
        self.escalation_triggered = false;
    }

    /// Get the self-correction configuration.
    ///
    /// TASK-METAUTL-P0-001: Provides access to constitution-defined parameters.
    pub fn config(&self) -> &SelfCorrectionConfig {
        &self.config
    }
}

use crate::protocol::{error_codes, methods, JsonRpcRequest, JsonRpcResponse};

/// Request handlers for MCP protocol.
///
/// Uses TeleologicalMemoryStore for 13-embedding fingerprint storage
/// and MultiArrayEmbeddingProvider for generating all 13 embeddings.
/// TASK-S003: Added GoalAlignmentCalculator and GoalHierarchy for purpose operations.
/// TASK-S004: Added JohariTransitionManager for johari/* operations.
/// TASK-S005: Added MetaUtlTracker for meta_utl/* operations.
/// TASK-EMB-024: Added SystemMonitor and LayerStatusProvider for real health metrics.
pub struct Handlers {
    /// Teleological memory store - stores TeleologicalFingerprint with 13 embeddings.
    /// NO legacy MemoryStore support.
    pub(super) teleological_store: Arc<dyn TeleologicalMemoryStore>,

    /// UTL processor for computing learning metrics.
    pub(super) utl_processor: Arc<dyn UtlProcessor>,

    /// Multi-array embedding provider - generates all 13 embeddings per content.
    /// NO legacy single-embedding support.
    pub(super) multi_array_provider: Arc<dyn MultiArrayEmbeddingProvider>,

    /// Goal alignment calculator - computes alignment between fingerprints and goal hierarchy.
    /// TASK-S003: Required for purpose/north_star_alignment and purpose/drift_check.
    /// TASK-INTEG-005: Will be used for cross-goal alignment calculations.
    #[allow(dead_code)]
    pub(super) alignment_calculator: Arc<dyn GoalAlignmentCalculator>,

    /// Goal hierarchy - defines North Star and sub-goals.
    /// TASK-S003: RwLock allows runtime updates via purpose/north_star_update.
    pub(super) goal_hierarchy: Arc<RwLock<GoalHierarchy>>,

    /// Johari transition manager - manages Johari quadrant transitions.
    /// TASK-S004: Required for johari/* handlers.
    pub(super) johari_manager: Arc<dyn JohariTransitionManager>,

    /// Meta-UTL tracker - tracks predictions and per-embedder accuracy.
    /// TASK-S005: Required for meta_utl/* handlers.
    pub(super) meta_utl_tracker: Arc<RwLock<MetaUtlTracker>>,

    /// System monitor for REAL health metrics.
    /// TASK-EMB-024: Required for meta_utl/health_metrics - NO hardcoded values.
    pub(super) system_monitor: Arc<dyn SystemMonitor>,

    /// Layer status provider for REAL layer statuses.
    /// TASK-EMB-024: Required for get_memetic_status and get_graph_manifest - NO hardcoded values.
    pub(super) layer_status_provider: Arc<dyn LayerStatusProvider>,

    // ========== GWT/Kuramoto Fields (TASK-GWT-001) ==========
    /// Kuramoto oscillator network for 13-embedding phase synchronization.
    /// TASK-GWT-001: Required for gwt/* handlers and consciousness computation.
    /// Uses RwLock because step() mutates internal state.
    pub(super) kuramoto_network: Option<Arc<RwLock<dyn KuramotoProvider>>>,

    /// GWT consciousness system provider.
    /// TASK-GWT-001: Required for consciousness computation C(t) = I(t) x R(t) x D(t).
    pub(super) gwt_system: Option<Arc<dyn GwtSystemProvider>>,

    /// Global workspace provider for winner-take-all memory selection.
    /// TASK-GWT-001: Required for workspace broadcast operations.
    pub(super) workspace_provider: Option<Arc<tokio::sync::RwLock<dyn WorkspaceProvider>>>,

    /// Meta-cognitive loop provider for self-correction.
    /// TASK-GWT-001: Required for meta_score computation and dream triggering.
    pub(super) meta_cognitive: Option<Arc<tokio::sync::RwLock<dyn MetaCognitiveProvider>>>,

    /// Self-ego node provider for system identity tracking.
    /// TASK-GWT-001: Required for identity continuity monitoring.
    pub(super) self_ego: Option<Arc<tokio::sync::RwLock<dyn SelfEgoProvider>>>,

    // ========== ADAPTIVE THRESHOLD CALIBRATION (TASK-ATC-001) ==========
    /// Adaptive Threshold Calibration system for self-learning thresholds.
    /// TASK-ATC-001: Required for get_threshold_status, get_calibration_metrics, trigger_recalibration.
    /// Uses RwLock because calibration operations mutate internal state.
    pub(super) atc: Option<Arc<RwLock<context_graph_core::atc::AdaptiveThresholdCalibration>>>,

    // ========== DREAM CONSOLIDATION (TASK-DREAM-MCP) ==========
    /// Dream controller for managing dream consolidation cycles.
    /// TASK-DREAM-MCP: Required for trigger_dream, get_dream_status, abort_dream.
    /// Uses RwLock because dream cycle operations mutate internal state.
    pub(super) dream_controller: Option<Arc<RwLock<context_graph_core::dream::DreamController>>>,

    /// Dream scheduler for determining when to trigger dream cycles.
    /// TASK-DREAM-MCP: Required for trigger_dream, get_dream_status.
    /// Uses RwLock because activity tracking mutates internal state.
    pub(super) dream_scheduler: Option<Arc<RwLock<context_graph_core::dream::DreamScheduler>>>,

    /// Amortized learner for shortcut creation during dreams.
    /// TASK-DREAM-MCP: Required for get_amortized_shortcuts.
    /// Uses RwLock because shortcut tracking mutates internal state.
    pub(super) amortized_learner: Option<Arc<RwLock<context_graph_core::dream::AmortizedLearner>>>,

    // ========== NEUROMODULATION (TASK-NEUROMOD-MCP) ==========
    /// Neuromodulation manager for controlling system behavior modulation.
    /// TASK-NEUROMOD-MCP: Required for get_neuromodulation_state, adjust_neuromodulator.
    /// Uses RwLock because modulator adjustments mutate internal state.
    pub(super) neuromod_manager:
        Option<Arc<RwLock<context_graph_core::neuromod::NeuromodulationManager>>>,
}

impl Handlers {
    /// Create new handlers with teleological dependencies.
    ///
    /// # Arguments
    /// * `teleological_store` - Store for TeleologicalFingerprint (TASK-F008)
    /// * `utl_processor` - UTL metrics computation
    /// * `multi_array_provider` - 13-embedding generator (TASK-F007)
    /// * `alignment_calculator` - Goal alignment calculator (TASK-S003)
    /// * `goal_hierarchy` - Goal hierarchy with North Star (TASK-S003)
    ///
    /// # TASK-EMB-024 Note
    ///
    /// This constructor uses StubSystemMonitor and StubLayerStatusProvider as defaults.
    /// For production use with real metrics, use `with_full_monitoring()`.
    #[allow(dead_code)]
    pub fn new(
        teleological_store: Arc<dyn TeleologicalMemoryStore>,
        utl_processor: Arc<dyn UtlProcessor>,
        multi_array_provider: Arc<dyn MultiArrayEmbeddingProvider>,
        alignment_calculator: Arc<dyn GoalAlignmentCalculator>,
        goal_hierarchy: GoalHierarchy,
    ) -> Self {
        // TASK-S004: Create Johari manager from teleological store
        let johari_manager: Arc<dyn JohariTransitionManager> =
            Arc::new(DynDefaultJohariManager::new(teleological_store.clone()));

        // TASK-S005: Create Meta-UTL tracker
        let meta_utl_tracker = Arc::new(RwLock::new(MetaUtlTracker::new()));

        // TASK-EMB-024: Default to stub monitors (will fail with explicit errors)
        let system_monitor: Arc<dyn SystemMonitor> = Arc::new(StubSystemMonitor::new());
        let layer_status_provider: Arc<dyn LayerStatusProvider> =
            Arc::new(StubLayerStatusProvider::new());

        Self {
            teleological_store,
            utl_processor,
            multi_array_provider,
            alignment_calculator,
            goal_hierarchy: Arc::new(RwLock::new(goal_hierarchy)),
            johari_manager,
            meta_utl_tracker,
            system_monitor,
            layer_status_provider,
            // TASK-GWT-001: GWT fields default to None - use with_gwt() for full GWT support
            kuramoto_network: None,
            gwt_system: None,
            workspace_provider: None,
            meta_cognitive: None,
            self_ego: None,
            // TASK-ATC-001: ATC defaults to None - use with_atc() for full ATC support
            atc: None,
            // TASK-DREAM-MCP: Dream fields default to None - use with_dream() for full dream support
            dream_controller: None,
            dream_scheduler: None,
            amortized_learner: None,
            // TASK-NEUROMOD-MCP: Neuromod defaults to None - use with_neuromod() for full support
            neuromod_manager: None,
        }
    }

    /// Create new handlers with shared goal hierarchy reference.
    ///
    /// Use this variant when you need to share the goal hierarchy across
    /// multiple handler instances or access it from outside.
    ///
    /// # Arguments
    /// * `teleological_store` - Store for TeleologicalFingerprint (TASK-F008)
    /// * `utl_processor` - UTL metrics computation
    /// * `multi_array_provider` - 13-embedding generator (TASK-F007)
    /// * `alignment_calculator` - Goal alignment calculator (TASK-S003)
    /// * `goal_hierarchy` - Shared goal hierarchy reference (TASK-S003)
    ///
    /// # TASK-EMB-024 Note
    ///
    /// This constructor uses StubSystemMonitor and StubLayerStatusProvider as defaults.
    #[allow(dead_code)]
    pub fn with_shared_hierarchy(
        teleological_store: Arc<dyn TeleologicalMemoryStore>,
        utl_processor: Arc<dyn UtlProcessor>,
        multi_array_provider: Arc<dyn MultiArrayEmbeddingProvider>,
        alignment_calculator: Arc<dyn GoalAlignmentCalculator>,
        goal_hierarchy: Arc<RwLock<GoalHierarchy>>,
    ) -> Self {
        // TASK-S004: Create Johari manager from teleological store
        let johari_manager: Arc<dyn JohariTransitionManager> =
            Arc::new(DynDefaultJohariManager::new(teleological_store.clone()));

        // TASK-S005: Create Meta-UTL tracker
        let meta_utl_tracker = Arc::new(RwLock::new(MetaUtlTracker::new()));

        // TASK-EMB-024: Default to stub monitors (will fail with explicit errors)
        let system_monitor: Arc<dyn SystemMonitor> = Arc::new(StubSystemMonitor::new());
        let layer_status_provider: Arc<dyn LayerStatusProvider> =
            Arc::new(StubLayerStatusProvider::new());

        Self {
            teleological_store,
            utl_processor,
            multi_array_provider,
            alignment_calculator,
            goal_hierarchy,
            johari_manager,
            meta_utl_tracker,
            system_monitor,
            layer_status_provider,
            // TASK-GWT-001: GWT fields default to None - use with_gwt() for full GWT support
            kuramoto_network: None,
            gwt_system: None,
            workspace_provider: None,
            meta_cognitive: None,
            self_ego: None,
            atc: None,
            // TASK-DREAM-MCP: Dream fields default to None
            dream_controller: None,
            dream_scheduler: None,
            amortized_learner: None,
            // TASK-NEUROMOD-MCP: Neuromod defaults to None
            neuromod_manager: None,
        }
    }

    /// Create new handlers with explicit Johari manager.
    ///
    /// Use this variant when you need to provide a custom JohariTransitionManager
    /// implementation or share it across multiple handler instances.
    ///
    /// # Arguments
    /// * `teleological_store` - Store for TeleologicalFingerprint (TASK-F008)
    /// * `utl_processor` - UTL metrics computation
    /// * `multi_array_provider` - 13-embedding generator (TASK-F007)
    /// * `alignment_calculator` - Goal alignment calculator (TASK-S003)
    /// * `goal_hierarchy` - Shared goal hierarchy reference (TASK-S003)
    /// * `johari_manager` - Shared Johari manager reference (TASK-S004)
    ///
    /// # TASK-EMB-024 Note
    ///
    /// This constructor uses StubSystemMonitor and StubLayerStatusProvider as defaults.
    #[allow(dead_code)]
    pub fn with_johari_manager(
        teleological_store: Arc<dyn TeleologicalMemoryStore>,
        utl_processor: Arc<dyn UtlProcessor>,
        multi_array_provider: Arc<dyn MultiArrayEmbeddingProvider>,
        alignment_calculator: Arc<dyn GoalAlignmentCalculator>,
        goal_hierarchy: Arc<RwLock<GoalHierarchy>>,
        johari_manager: Arc<dyn JohariTransitionManager>,
    ) -> Self {
        // TASK-S005: Create Meta-UTL tracker
        let meta_utl_tracker = Arc::new(RwLock::new(MetaUtlTracker::new()));

        // TASK-EMB-024: Default to stub monitors (will fail with explicit errors)
        let system_monitor: Arc<dyn SystemMonitor> = Arc::new(StubSystemMonitor::new());
        let layer_status_provider: Arc<dyn LayerStatusProvider> =
            Arc::new(StubLayerStatusProvider::new());

        Self {
            teleological_store,
            utl_processor,
            multi_array_provider,
            alignment_calculator,
            goal_hierarchy,
            johari_manager,
            meta_utl_tracker,
            system_monitor,
            layer_status_provider,
            // TASK-GWT-001: GWT fields default to None - use with_gwt() for full GWT support
            kuramoto_network: None,
            gwt_system: None,
            workspace_provider: None,
            meta_cognitive: None,
            self_ego: None,
            atc: None,
            // TASK-DREAM-MCP: Dream fields default to None
            dream_controller: None,
            dream_scheduler: None,
            amortized_learner: None,
            // TASK-NEUROMOD-MCP: Neuromod defaults to None
            neuromod_manager: None,
        }
    }

    /// Create new handlers with explicit Meta-UTL tracker.
    ///
    /// Use this variant when you need to provide a custom MetaUtlTracker
    /// implementation or share it across multiple handler instances (for testing).
    ///
    /// TASK-S005: Added for full state verification tests.
    ///
    /// # TASK-EMB-024 Note
    ///
    /// This constructor uses StubSystemMonitor and StubLayerStatusProvider as defaults.
    #[allow(dead_code)]
    pub fn with_meta_utl_tracker(
        teleological_store: Arc<dyn TeleologicalMemoryStore>,
        utl_processor: Arc<dyn UtlProcessor>,
        multi_array_provider: Arc<dyn MultiArrayEmbeddingProvider>,
        alignment_calculator: Arc<dyn GoalAlignmentCalculator>,
        goal_hierarchy: Arc<RwLock<GoalHierarchy>>,
        johari_manager: Arc<dyn JohariTransitionManager>,
        meta_utl_tracker: Arc<RwLock<MetaUtlTracker>>,
    ) -> Self {
        // TASK-EMB-024: Default to stub monitors (will fail with explicit errors)
        let system_monitor: Arc<dyn SystemMonitor> = Arc::new(StubSystemMonitor::new());
        let layer_status_provider: Arc<dyn LayerStatusProvider> =
            Arc::new(StubLayerStatusProvider::new());

        Self {
            teleological_store,
            utl_processor,
            multi_array_provider,
            alignment_calculator,
            goal_hierarchy,
            johari_manager,
            meta_utl_tracker,
            system_monitor,
            layer_status_provider,
            // TASK-GWT-001: GWT fields default to None - use with_gwt() for full GWT support
            kuramoto_network: None,
            gwt_system: None,
            workspace_provider: None,
            meta_cognitive: None,
            self_ego: None,
            atc: None,
            // TASK-DREAM-MCP: Dream fields default to None
            dream_controller: None,
            dream_scheduler: None,
            amortized_learner: None,
            // TASK-NEUROMOD-MCP: Neuromod defaults to None
            neuromod_manager: None,
        }
    }

    /// Create new handlers with full monitoring support.
    ///
    /// TASK-EMB-024: This is the recommended constructor for production use
    /// when you need REAL health metrics (no hardcoded values).
    ///
    /// # Arguments
    /// * `teleological_store` - Store for TeleologicalFingerprint (TASK-F008)
    /// * `utl_processor` - UTL metrics computation
    /// * `multi_array_provider` - 13-embedding generator (TASK-F007)
    /// * `alignment_calculator` - Goal alignment calculator (TASK-S003)
    /// * `goal_hierarchy` - Shared goal hierarchy reference (TASK-S003)
    /// * `johari_manager` - Shared Johari manager reference (TASK-S004)
    /// * `meta_utl_tracker` - Shared Meta-UTL tracker (TASK-S005)
    /// * `system_monitor` - Real system monitor for health metrics
    /// * `layer_status_provider` - Real layer status provider
    #[allow(dead_code)]
    #[allow(clippy::too_many_arguments)]
    pub fn with_full_monitoring(
        teleological_store: Arc<dyn TeleologicalMemoryStore>,
        utl_processor: Arc<dyn UtlProcessor>,
        multi_array_provider: Arc<dyn MultiArrayEmbeddingProvider>,
        alignment_calculator: Arc<dyn GoalAlignmentCalculator>,
        goal_hierarchy: Arc<RwLock<GoalHierarchy>>,
        johari_manager: Arc<dyn JohariTransitionManager>,
        meta_utl_tracker: Arc<RwLock<MetaUtlTracker>>,
        system_monitor: Arc<dyn SystemMonitor>,
        layer_status_provider: Arc<dyn LayerStatusProvider>,
    ) -> Self {
        Self {
            teleological_store,
            utl_processor,
            multi_array_provider,
            alignment_calculator,
            goal_hierarchy,
            johari_manager,
            meta_utl_tracker,
            system_monitor,
            layer_status_provider,
            // TASK-GWT-001: GWT fields default to None - use with_gwt() for full GWT support
            kuramoto_network: None,
            gwt_system: None,
            workspace_provider: None,
            meta_cognitive: None,
            self_ego: None,
            // TASK-ATC-001: ATC provider default to None - use with_atc() for ATC support
            atc: None,
            // TASK-DREAM-MCP: Dream fields default to None
            dream_controller: None,
            dream_scheduler: None,
            amortized_learner: None,
            // TASK-NEUROMOD-MCP: Neuromod defaults to None
            neuromod_manager: None,
        }
    }

    /// Create new handlers with full GWT/consciousness support.
    ///
    /// TASK-GWT-001: This is the recommended constructor for production use
    /// with REAL GWT consciousness features. All GWT providers are REQUIRED.
    /// No stub implementations allowed - FAIL FAST on missing components.
    ///
    /// # Arguments
    /// * `teleological_store` - Store for TeleologicalFingerprint (TASK-F008)
    /// * `utl_processor` - UTL metrics computation
    /// * `multi_array_provider` - 13-embedding generator (TASK-F007)
    /// * `alignment_calculator` - Goal alignment calculator (TASK-S003)
    /// * `goal_hierarchy` - Shared goal hierarchy reference (TASK-S003)
    /// * `johari_manager` - Shared Johari manager reference (TASK-S004)
    /// * `meta_utl_tracker` - Shared Meta-UTL tracker (TASK-S005)
    /// * `system_monitor` - Real system monitor for health metrics
    /// * `layer_status_provider` - Real layer status provider
    /// * `kuramoto_network` - Kuramoto oscillator network (TASK-GWT-001)
    /// * `gwt_system` - GWT consciousness system (TASK-GWT-001)
    /// * `workspace_provider` - Global workspace provider (TASK-GWT-001)
    /// * `meta_cognitive` - Meta-cognitive loop provider (TASK-GWT-001)
    /// * `self_ego` - Self-ego node provider (TASK-GWT-001)
    #[allow(clippy::too_many_arguments)]
    #[allow(dead_code)]
    pub fn with_gwt(
        teleological_store: Arc<dyn TeleologicalMemoryStore>,
        utl_processor: Arc<dyn UtlProcessor>,
        multi_array_provider: Arc<dyn MultiArrayEmbeddingProvider>,
        alignment_calculator: Arc<dyn GoalAlignmentCalculator>,
        goal_hierarchy: Arc<RwLock<GoalHierarchy>>,
        johari_manager: Arc<dyn JohariTransitionManager>,
        meta_utl_tracker: Arc<RwLock<MetaUtlTracker>>,
        system_monitor: Arc<dyn SystemMonitor>,
        layer_status_provider: Arc<dyn LayerStatusProvider>,
        kuramoto_network: Arc<RwLock<dyn KuramotoProvider>>,
        gwt_system: Arc<dyn GwtSystemProvider>,
        workspace_provider: Arc<tokio::sync::RwLock<dyn WorkspaceProvider>>,
        meta_cognitive: Arc<tokio::sync::RwLock<dyn MetaCognitiveProvider>>,
        self_ego: Arc<tokio::sync::RwLock<dyn SelfEgoProvider>>,
    ) -> Self {
        Self {
            teleological_store,
            utl_processor,
            multi_array_provider,
            alignment_calculator,
            goal_hierarchy,
            johari_manager,
            meta_utl_tracker,
            system_monitor,
            layer_status_provider,
            kuramoto_network: Some(kuramoto_network),
            gwt_system: Some(gwt_system),
            workspace_provider: Some(workspace_provider),
            meta_cognitive: Some(meta_cognitive),
            self_ego: Some(self_ego),
            // TASK-ATC-001: ATC provider default to None - use with_atc() for ATC support
            atc: None,
            // TASK-DREAM-MCP: Dream fields default to None
            dream_controller: None,
            dream_scheduler: None,
            amortized_learner: None,
            // TASK-NEUROMOD-MCP: Neuromod defaults to None
            neuromod_manager: None,
        }
    }

    /// Create new handlers with ALL subsystems wired (GWT + ATC + Dream + Neuromod).
    ///
    /// TASK-EXHAUSTIVE-MCP: This is the ONLY constructor that enables ALL 35 MCP tools.
    /// Use for:
    /// - Exhaustive MCP tool testing
    /// - Full system integration tests
    /// - Production deployment where all features are required
    ///
    /// # Arguments
    /// All GWT providers (same as with_gwt) plus:
    /// * `atc` - Adaptive Threshold Calibration for get_threshold_status, get_calibration_metrics, trigger_recalibration
    /// * `dream_controller` - Dream consolidation controller for trigger_dream, get_dream_status, abort_dream
    /// * `dream_scheduler` - Dream scheduling logic
    /// * `amortized_learner` - Amortized shortcut learner for get_amortized_shortcuts
    /// * `neuromod_manager` - Neuromodulation manager for get_neuromodulation_state, adjust_neuromodulator
    #[allow(clippy::too_many_arguments)]
    pub fn with_gwt_and_subsystems(
        teleological_store: Arc<dyn TeleologicalMemoryStore>,
        utl_processor: Arc<dyn UtlProcessor>,
        multi_array_provider: Arc<dyn MultiArrayEmbeddingProvider>,
        alignment_calculator: Arc<dyn GoalAlignmentCalculator>,
        goal_hierarchy: Arc<RwLock<GoalHierarchy>>,
        johari_manager: Arc<dyn JohariTransitionManager>,
        meta_utl_tracker: Arc<RwLock<MetaUtlTracker>>,
        system_monitor: Arc<dyn SystemMonitor>,
        layer_status_provider: Arc<dyn LayerStatusProvider>,
        kuramoto_network: Arc<RwLock<dyn KuramotoProvider>>,
        gwt_system: Arc<dyn GwtSystemProvider>,
        workspace_provider: Arc<tokio::sync::RwLock<dyn WorkspaceProvider>>,
        meta_cognitive: Arc<tokio::sync::RwLock<dyn MetaCognitiveProvider>>,
        self_ego: Arc<tokio::sync::RwLock<dyn SelfEgoProvider>>,
        atc: Arc<RwLock<context_graph_core::atc::AdaptiveThresholdCalibration>>,
        dream_controller: Arc<RwLock<context_graph_core::dream::DreamController>>,
        dream_scheduler: Arc<RwLock<context_graph_core::dream::DreamScheduler>>,
        amortized_learner: Arc<RwLock<context_graph_core::dream::AmortizedLearner>>,
        neuromod_manager: Arc<RwLock<context_graph_core::neuromod::NeuromodulationManager>>,
    ) -> Self {
        Self {
            teleological_store,
            utl_processor,
            multi_array_provider,
            alignment_calculator,
            goal_hierarchy,
            johari_manager,
            meta_utl_tracker,
            system_monitor,
            layer_status_provider,
            kuramoto_network: Some(kuramoto_network),
            gwt_system: Some(gwt_system),
            workspace_provider: Some(workspace_provider),
            meta_cognitive: Some(meta_cognitive),
            self_ego: Some(self_ego),
            atc: Some(atc),
            dream_controller: Some(dream_controller),
            dream_scheduler: Some(dream_scheduler),
            amortized_learner: Some(amortized_learner),
            neuromod_manager: Some(neuromod_manager),
        }
    }

    /// Create new handlers with default GWT provider implementations.
    ///
    /// TASK-GWT-001: Convenience constructor that uses the real GWT provider
    /// implementations from `gwt_providers` module. This creates fresh instances
    /// of KuramotoNetwork, ConsciousnessCalculator, GlobalWorkspace, etc.
    ///
    /// All GWT tools will return REAL data - no stubs, no mocks.
    ///
    /// # Arguments
    /// * `teleological_store` - Store for TeleologicalFingerprint (TASK-F008)
    /// * `utl_processor` - UTL metrics computation
    /// * `multi_array_provider` - 13-embedding generator (TASK-F007)
    /// * `alignment_calculator` - Goal alignment calculator (TASK-S003)
    /// * `goal_hierarchy` - Shared goal hierarchy reference (TASK-S003)
    /// * `johari_manager` - Shared Johari manager reference (TASK-S004)
    /// * `meta_utl_tracker` - Shared Meta-UTL tracker (TASK-S005)
    /// * `system_monitor` - Real system monitor for health metrics
    /// * `layer_status_provider` - Real layer status provider
    #[allow(clippy::too_many_arguments)]
    pub fn with_default_gwt(
        teleological_store: Arc<dyn TeleologicalMemoryStore>,
        utl_processor: Arc<dyn UtlProcessor>,
        multi_array_provider: Arc<dyn MultiArrayEmbeddingProvider>,
        alignment_calculator: Arc<dyn GoalAlignmentCalculator>,
        goal_hierarchy: Arc<RwLock<GoalHierarchy>>,
        johari_manager: Arc<dyn JohariTransitionManager>,
        meta_utl_tracker: Arc<RwLock<MetaUtlTracker>>,
        system_monitor: Arc<dyn SystemMonitor>,
        layer_status_provider: Arc<dyn LayerStatusProvider>,
    ) -> Self {
        use super::gwt_providers::{
            GwtSystemProviderImpl, KuramotoProviderImpl, MetaCognitiveProviderImpl,
            SelfEgoProviderImpl, WorkspaceProviderImpl,
        };

        // Create real GWT provider implementations
        let kuramoto_network: Arc<RwLock<dyn KuramotoProvider>> =
            Arc::new(RwLock::new(KuramotoProviderImpl::new()));
        let gwt_system: Arc<dyn GwtSystemProvider> = Arc::new(GwtSystemProviderImpl::new());
        let workspace_provider: Arc<tokio::sync::RwLock<dyn WorkspaceProvider>> =
            Arc::new(tokio::sync::RwLock::new(WorkspaceProviderImpl::new()));
        let meta_cognitive: Arc<tokio::sync::RwLock<dyn MetaCognitiveProvider>> =
            Arc::new(tokio::sync::RwLock::new(MetaCognitiveProviderImpl::new()));
        let self_ego: Arc<tokio::sync::RwLock<dyn SelfEgoProvider>> =
            Arc::new(tokio::sync::RwLock::new(SelfEgoProviderImpl::new()));

        // TASK-NEUROMOD-MCP: Create REAL NeuromodulationManager with default baselines
        // Constitution neuromod section: Dopamine [1,5], Serotonin [0,1], Noradrenaline [0.5,2]
        // Acetylcholine is READ-ONLY, managed by GWT MetaCognitiveLoop
        let neuromod_manager: Arc<RwLock<NeuromodulationManager>> =
            Arc::new(RwLock::new(NeuromodulationManager::new()));

        // TASK-DREAM-MCP: Create REAL Dream components with constitution-mandated defaults
        // Constitution dream section:
        // - Trigger: activity < 0.15, idle 10min
        // - NREM: 3min, replay recent, tight coupling, recency_bias 0.8
        // - REM: 2min, explore attractors, temp 2.0
        // - Constraints: 100 queries, semantic_leap 0.7, abort_on_query, wake <100ms, gpu <30%
        // - Amortized: 3+ hop ≥5×, weight product(path), confidence ≥0.7
        let dream_controller: Arc<RwLock<DreamController>> =
            Arc::new(RwLock::new(DreamController::new()));
        let dream_scheduler: Arc<RwLock<DreamScheduler>> =
            Arc::new(RwLock::new(DreamScheduler::new()));
        let amortized_learner: Arc<RwLock<AmortizedLearner>> =
            Arc::new(RwLock::new(AmortizedLearner::new()));

        // TASK-ATC-001: Create REAL AdaptiveThresholdCalibration with constitution-mandated defaults
        // Constitution adaptive_thresholds section:
        // - Level 1 EWMA Drift Tracker (per-query)
        // - Level 2 Temperature Scaling (hourly, per-embedder T values)
        // - Level 3 Bandit Threshold Selector (session, UCB/Thompson Sampling)
        // - Level 4 Bayesian Meta-Optimizer (weekly, GP surrogate + EI acquisition)
        // Threshold priors: θ_opt=0.75, θ_acc=0.70, θ_warn=0.55, θ_dup=0.90, θ_edge=0.70, etc.
        let atc: Arc<RwLock<AdaptiveThresholdCalibration>> =
            Arc::new(RwLock::new(AdaptiveThresholdCalibration::new()));

        Self {
            teleological_store,
            utl_processor,
            multi_array_provider,
            alignment_calculator,
            goal_hierarchy,
            johari_manager,
            meta_utl_tracker,
            system_monitor,
            layer_status_provider,
            kuramoto_network: Some(kuramoto_network),
            gwt_system: Some(gwt_system),
            workspace_provider: Some(workspace_provider),
            meta_cognitive: Some(meta_cognitive),
            self_ego: Some(self_ego),
            // TASK-ATC-001: REAL AdaptiveThresholdCalibration wired
            atc: Some(atc),
            // TASK-DREAM-MCP: REAL Dream components wired
            dream_controller: Some(dream_controller),
            dream_scheduler: Some(dream_scheduler),
            amortized_learner: Some(amortized_learner),
            // TASK-NEUROMOD-MCP: REAL NeuromodulationManager wired
            neuromod_manager: Some(neuromod_manager),
        }
    }

    /// Dispatch a request to the appropriate handler.
    pub async fn dispatch(&self, request: JsonRpcRequest) -> JsonRpcResponse {
        debug!("Dispatching method: {}", request.method);

        match request.method.as_str() {
            // MCP lifecycle methods
            methods::INITIALIZE => self.handle_initialize(request.id).await,
            "notifications/initialized" => self.handle_initialized_notification(),
            methods::SHUTDOWN => self.handle_shutdown(request.id).await,

            // MCP tools protocol
            methods::TOOLS_LIST => self.handle_tools_list(request.id).await,
            methods::TOOLS_CALL => self.handle_tools_call(request.id, request.params).await,

            // Legacy direct methods (kept for backward compatibility)
            methods::MEMORY_STORE => self.handle_memory_store(request.id, request.params).await,
            methods::MEMORY_RETRIEVE => {
                self.handle_memory_retrieve(request.id, request.params)
                    .await
            }
            methods::MEMORY_SEARCH => self.handle_memory_search(request.id, request.params).await,
            methods::MEMORY_DELETE => self.handle_memory_delete(request.id, request.params).await,

            // Memory injection and comparison operations (TASK-INTEG-001)
            methods::MEMORY_INJECT => self.handle_memory_inject(request.id, request.params).await,
            methods::MEMORY_INJECT_BATCH => {
                self.handle_memory_inject_batch(request.id, request.params)
                    .await
            }
            methods::MEMORY_SEARCH_MULTI_PERSPECTIVE => {
                self.handle_memory_search_multi_perspective(request.id, request.params)
                    .await
            }
            methods::MEMORY_COMPARE => {
                self.handle_memory_compare(request.id, request.params)
                    .await
            }
            methods::MEMORY_BATCH_COMPARE => {
                self.handle_memory_batch_compare(request.id, request.params)
                    .await
            }
            methods::MEMORY_SIMILARITY_MATRIX => {
                self.handle_memory_similarity_matrix(request.id, request.params)
                    .await
            }

            // Search operations (TASK-S002)
            methods::SEARCH_MULTI => self.handle_search_multi(request.id, request.params).await,
            methods::SEARCH_SINGLE_SPACE => {
                self.handle_search_single_space(request.id, request.params)
                    .await
            }
            methods::SEARCH_BY_PURPOSE => {
                self.handle_search_by_purpose(request.id, request.params)
                    .await
            }
            methods::SEARCH_WEIGHT_PROFILES => self.handle_get_weight_profiles(request.id).await,

            // Purpose/goal operations (TASK-S003)
            // NOTE: PURPOSE_NORTH_STAR_ALIGNMENT and NORTH_STAR_UPDATE removed per TASK-CORE-001 (ARCH-03)
            // These methods now fall through to the default case returning METHOD_NOT_FOUND (-32601)
            // Use auto_bootstrap_north_star tool for autonomous goal discovery instead.
            methods::PURPOSE_QUERY => self.handle_purpose_query(request.id, request.params).await,
            methods::GOAL_HIERARCHY_QUERY => {
                self.handle_goal_hierarchy_query(request.id, request.params)
                    .await
            }
            methods::GOAL_ALIGNED_MEMORIES => {
                self.handle_goal_aligned_memories(request.id, request.params)
                    .await
            }
            methods::PURPOSE_DRIFT_CHECK => {
                self.handle_purpose_drift_check(request.id, request.params)
                    .await
            }

            // Johari operations (TASK-S004)
            methods::JOHARI_GET_DISTRIBUTION => {
                self.handle_johari_get_distribution(request.id, request.params)
                    .await
            }
            methods::JOHARI_FIND_BY_QUADRANT => {
                self.handle_johari_find_by_quadrant(request.id, request.params)
                    .await
            }
            methods::JOHARI_TRANSITION => {
                self.handle_johari_transition(request.id, request.params)
                    .await
            }
            methods::JOHARI_TRANSITION_BATCH => {
                self.handle_johari_transition_batch(request.id, request.params)
                    .await
            }
            methods::JOHARI_CROSS_SPACE_ANALYSIS => {
                self.handle_johari_cross_space_analysis(request.id, request.params)
                    .await
            }
            methods::JOHARI_TRANSITION_PROBABILITIES => {
                self.handle_johari_transition_probabilities(request.id, request.params)
                    .await
            }

            methods::UTL_COMPUTE => self.handle_utl_compute(request.id, request.params).await,
            methods::UTL_METRICS => self.handle_utl_metrics(request.id, request.params).await,

            // Meta-UTL operations (TASK-S005)
            methods::META_UTL_LEARNING_TRAJECTORY => {
                self.handle_meta_utl_learning_trajectory(request.id, request.params)
                    .await
            }
            methods::META_UTL_HEALTH_METRICS => {
                self.handle_meta_utl_health_metrics(request.id, request.params)
                    .await
            }
            methods::META_UTL_PREDICT_STORAGE => {
                self.handle_meta_utl_predict_storage(request.id, request.params)
                    .await
            }
            methods::META_UTL_PREDICT_RETRIEVAL => {
                self.handle_meta_utl_predict_retrieval(request.id, request.params)
                    .await
            }
            methods::META_UTL_VALIDATE_PREDICTION => {
                self.handle_meta_utl_validate_prediction(request.id, request.params)
                    .await
            }
            methods::META_UTL_OPTIMIZED_WEIGHTS => {
                self.handle_meta_utl_optimized_weights(request.id, request.params)
                    .await
            }

            // Consciousness JSON-RPC methods (TASK-INTEG-003)
            // These delegate to existing tool implementations for hook integration.
            methods::CONSCIOUSNESS_GET_STATE => {
                // Delegate to existing get_consciousness_state tool implementation
                self.call_get_consciousness_state(request.id).await
            }
            methods::CONSCIOUSNESS_SYNC_LEVEL => {
                // Delegate to existing get_kuramoto_sync tool implementation
                self.call_get_kuramoto_sync(request.id).await
            }

            methods::SYSTEM_STATUS => self.handle_system_status(request.id).await,
            methods::SYSTEM_HEALTH => self.handle_system_health(request.id).await,
            _ => JsonRpcResponse::error(
                request.id,
                error_codes::METHOD_NOT_FOUND,
                format!("Method not found: {}", request.method),
            ),
        }
    }
}
