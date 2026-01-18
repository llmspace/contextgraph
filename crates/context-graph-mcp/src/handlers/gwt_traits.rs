//! GWT Provider Traits for MCP Handler Integration
//!
//! TASK-GWT-001: Provider traits that wrap GWT components for handler injection.
//! All traits require Send + Sync for thread-safe Arc wrapping.
//!
//! These traits define the interface for GWT components to be wired
//! into the MCP Handlers struct. All implementations must use REAL components -
//! NO STUBS, NO FALLBACKS, FAIL FAST on errors.
//!
//! # Note on Topic Stability
//!
//! Per PRD v6 Section 14, topic-based coherence scoring is used for session
//! health. Use `get_topic_stability()` on WorkspaceProvider for health checks.

use std::time::Duration;

use async_trait::async_trait;
use context_graph_core::error::CoreResult;
use context_graph_core::gwt::{MetaCognitiveState, StateTransition};
use uuid::Uuid;

/// Provider trait for GWT state management.
///
/// Wraps StateMachineManager from context-graph-core for handler access.
/// TASK-GWT-001: Required for GWT state tracking.
///
/// # Health Thresholds
///
/// Use `WorkspaceProvider::get_topic_stability()` for health checks.
/// Health thresholds per constitution topic_stability.thresholds:
/// - healthy: churn < 0.3 (topic_stability >= 0.7)
/// - warning: churn in [0.3, 0.5) (topic_stability in [0.5, 0.7))
/// - critical: churn >= 0.5 (topic_stability < 0.5)
#[async_trait]
pub trait GwtSystemProvider: Send + Sync {
    /// Check if system is in a coherent state (topic_stability >= 0.7).
    ///
    /// Per PRD v6, this uses topic-based coherence scoring.
    #[allow(dead_code)]
    fn is_coherent(&self) -> bool;

    /// Get the last state transition if any.
    #[allow(dead_code)]
    fn last_transition(&self) -> Option<StateTransition>;

    /// Get time spent in current state.
    fn time_in_state(&self) -> Duration;
}

/// Provider trait for workspace selection operations.
///
/// Handles winner-take-all memory selection for global workspace.
/// All methods are async to prevent deadlock with single-threaded runtimes.
///
/// TASK-GWT-001: Required for workspace broadcast operations.
/// Constitution: AP-08 ("No sync I/O in async context")
#[async_trait]
pub trait WorkspaceProvider: Send + Sync {
    /// Select winning memory via winner-take-all algorithm.
    ///
    /// # Arguments
    /// - candidates: Vec of (memory_id, order_parameter_r, importance, alignment)
    ///
    /// # Returns
    /// UUID of winning memory, or None if no candidates pass coherence threshold (0.8)
    async fn select_winning_memory(
        &self,
        candidates: Vec<(Uuid, f32, f32, f32)>,
    ) -> CoreResult<Option<Uuid>>;

    /// Get currently active (conscious) memory if broadcasting.
    async fn get_active_memory(&self) -> Option<Uuid>;

    /// Check if broadcast window is still active.
    async fn is_broadcasting(&self) -> bool;

    /// Check for workspace conflict (multiple memories with r > 0.8).
    async fn has_conflict(&self) -> bool;

    /// Get conflicting memory IDs if present.
    async fn get_conflict_details(&self) -> Option<Vec<Uuid>>;

    /// Get coherence threshold for workspace entry.
    async fn coherence_threshold(&self) -> f32;

    /// Get topic stability score [0, 1].
    async fn get_topic_stability(&self) -> f32;

    /// Get 13D purpose vector representing alignment to strategic goals.
    async fn get_purpose_vector(&self) -> Vec<f32>;
}

/// Provider trait for meta-cognitive loop operations.
///
/// All methods are async per Constitution AP-08: "No sync I/O in async context".
/// TASK-GWT-001: Required for meta_score computation and dream triggering.
#[async_trait]
pub trait MetaCognitiveProvider: Send + Sync {
    /// Evaluate meta-cognitive score.
    ///
    /// MetaScore = sigmoid(2 x (L_predicted - L_actual))
    ///
    /// # Arguments
    /// - predicted_learning: L_predicted in [0, 1]
    /// - actual_learning: L_actual in [0, 1]
    #[allow(dead_code)]
    async fn evaluate(
        &self,
        predicted_learning: f32,
        actual_learning: f32,
    ) -> CoreResult<MetaCognitiveState>;

    /// Get current Acetylcholine level (learning rate modulator).
    async fn acetylcholine(&self) -> f32;

    /// Get current monitoring frequency (Hz).
    #[allow(dead_code)]
    async fn monitoring_frequency(&self) -> f32;

    /// Get recent meta-scores for trend analysis.
    #[allow(dead_code)]
    async fn get_recent_scores(&self) -> Vec<f32>;
}
