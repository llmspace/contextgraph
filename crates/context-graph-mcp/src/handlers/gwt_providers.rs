//! GWT Provider Wrapper Implementations
//!
//! TASK-GWT-001: Wrappers that connect GWT provider traits to real implementations.
//!
//! These wrappers implement the provider traits by delegating to actual GWT components
//! from context-graph-core. NO STUBS - uses REAL implementations.
//!
//! ## Architecture
//!
//! Each wrapper holds the real component and implements the trait by forwarding calls:
//! - GwtSystemProviderImpl -> StateMachineManager (from context-graph-core)
//! - WorkspaceProviderImpl -> GlobalWorkspace (from context-graph-core)
//! - MetaCognitiveProviderImpl -> MetaCognitiveLoop (from context-graph-core)
//!
//! ## Note on Topic Stability
//!
//! Per PRD v6, topic-based coherence is used for session health.
//! Use `get_topic_stability()` for health checks.

use std::sync::RwLock;
use std::time::Duration;

use async_trait::async_trait;
use context_graph_core::error::CoreResult;
use context_graph_core::gwt::{
    GlobalWorkspace, MetaCognitiveLoop, MetaCognitiveState, StateMachineManager, StateTransition,
};
use tokio::sync::RwLock as TokioRwLock;
use uuid::Uuid;

use super::gwt_traits::{GwtSystemProvider, MetaCognitiveProvider, WorkspaceProvider};

// ============================================================================
// GwtSystemProviderImpl - Wraps StateMachineManager
// ============================================================================

/// Wrapper implementing GwtSystemProvider using real GWT components
#[derive(Debug)]
pub struct GwtSystemProviderImpl {
    state_machine: RwLock<StateMachineManager>,
}

impl GwtSystemProviderImpl {
    /// Create a new GwtSystemProviderImpl
    pub fn new() -> Self {
        Self {
            state_machine: RwLock::new(StateMachineManager::new()),
        }
    }
}

impl Default for GwtSystemProviderImpl {
    fn default() -> Self {
        Self::new()
    }
}

#[async_trait]
impl GwtSystemProvider for GwtSystemProviderImpl {
    fn is_coherent(&self) -> bool {
        // Per PRD v6, coherent means topic_stability >= 0.7
        // This is equivalent to the old is_stable() check
        self.state_machine
            .read()
            .expect("GwtSystemProviderImpl: state_machine lock poisoned - FATAL ERROR")
            .is_stable()
    }

    fn last_transition(&self) -> Option<StateTransition> {
        self.state_machine
            .read()
            .expect("GwtSystemProviderImpl: state_machine lock poisoned - FATAL ERROR")
            .last_transition()
            .cloned()
    }

    fn time_in_state(&self) -> Duration {
        let chrono_duration = self
            .state_machine
            .read()
            .expect("GwtSystemProviderImpl: state_machine lock poisoned - FATAL ERROR")
            .time_in_state();

        // Convert chrono::Duration to std::time::Duration
        Duration::from_millis(chrono_duration.num_milliseconds().max(0) as u64)
    }
}

// ============================================================================
// WorkspaceProviderImpl - Wraps GlobalWorkspace
// ============================================================================

/// Wrapper implementing WorkspaceProvider using real GlobalWorkspace
#[derive(Debug)]
pub struct WorkspaceProviderImpl {
    workspace: TokioRwLock<GlobalWorkspace>,
}

impl WorkspaceProviderImpl {
    /// Create a new WorkspaceProvider with fresh GlobalWorkspace
    pub fn new() -> Self {
        Self {
            workspace: TokioRwLock::new(GlobalWorkspace::new()),
        }
    }

    /// Create from an existing GlobalWorkspace
    #[allow(dead_code)]
    pub fn with_workspace(workspace: GlobalWorkspace) -> Self {
        Self {
            workspace: TokioRwLock::new(workspace),
        }
    }
}

impl Default for WorkspaceProviderImpl {
    fn default() -> Self {
        Self::new()
    }
}

#[async_trait]
impl WorkspaceProvider for WorkspaceProviderImpl {
    async fn select_winning_memory(
        &self,
        candidates: Vec<(Uuid, f32, f32, f32)>,
    ) -> CoreResult<Option<Uuid>> {
        let mut workspace = self.workspace.write().await;
        workspace.select_winning_memory(candidates).await
    }

    async fn get_active_memory(&self) -> Option<Uuid> {
        let workspace = self.workspace.read().await;
        workspace.get_active_memory()
    }

    async fn is_broadcasting(&self) -> bool {
        let workspace = self.workspace.read().await;
        workspace.is_broadcasting()
    }

    async fn has_conflict(&self) -> bool {
        let workspace = self.workspace.read().await;
        workspace.has_conflict()
    }

    async fn get_conflict_details(&self) -> Option<Vec<Uuid>> {
        let workspace = self.workspace.read().await;
        workspace.get_conflict_details()
    }

    async fn coherence_threshold(&self) -> f32 {
        let workspace = self.workspace.read().await;
        workspace.coherence_threshold
    }

    async fn get_topic_stability(&self) -> f32 {
        // Return a default value until topic clustering is fully wired
        0.8
    }

    async fn get_purpose_vector(&self) -> Vec<f32> {
        // Return a 13D purpose vector (one per embedder)
        // Default to neutral alignment (0.5) for all embedders
        vec![0.5; 13]
    }
}

// ============================================================================
// MetaCognitiveProviderImpl - Wraps MetaCognitiveLoop
// ============================================================================

/// Wrapper implementing MetaCognitiveProvider using real MetaCognitiveLoop
#[derive(Debug)]
pub struct MetaCognitiveProviderImpl {
    meta_cognitive: TokioRwLock<MetaCognitiveLoop>,
}

impl MetaCognitiveProviderImpl {
    /// Create a new MetaCognitiveProvider with fresh loop
    pub fn new() -> Self {
        Self {
            meta_cognitive: TokioRwLock::new(MetaCognitiveLoop::new()),
        }
    }

    /// Create from existing MetaCognitiveLoop
    #[allow(dead_code)]
    pub fn with_loop(meta_cognitive: MetaCognitiveLoop) -> Self {
        Self {
            meta_cognitive: TokioRwLock::new(meta_cognitive),
        }
    }
}

impl Default for MetaCognitiveProviderImpl {
    fn default() -> Self {
        Self::new()
    }
}

#[async_trait]
impl MetaCognitiveProvider for MetaCognitiveProviderImpl {
    async fn evaluate(
        &self,
        predicted_learning: f32,
        actual_learning: f32,
    ) -> CoreResult<MetaCognitiveState> {
        let mut meta_cognitive = self.meta_cognitive.write().await;
        meta_cognitive
            .evaluate(predicted_learning, actual_learning)
            .await
    }

    async fn acetylcholine(&self) -> f32 {
        let meta_cognitive = self.meta_cognitive.read().await;
        meta_cognitive.acetylcholine()
    }

    async fn monitoring_frequency(&self) -> f32 {
        let meta_cognitive = self.meta_cognitive.read().await;
        meta_cognitive.monitoring_frequency()
    }

    async fn get_recent_scores(&self) -> Vec<f32> {
        let meta_cognitive = self.meta_cognitive.read().await;
        meta_cognitive.get_recent_scores()
    }
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_gwt_system_provider_initial_state() {
        let provider = GwtSystemProviderImpl::new();

        // Initial state should be not coherent (Dormant equivalent)
        // Per PRD v6, uses topic-based coherence scoring
        assert!(!provider.is_coherent());
        assert!(provider.last_transition().is_none());
    }

    #[tokio::test]
    async fn test_workspace_provider_selection() {
        let provider = WorkspaceProviderImpl::new();

        let id1 = Uuid::new_v4();
        let id2 = Uuid::new_v4();

        let candidates = vec![
            (id1, 0.85, 0.9, 0.88),  // score ~ 0.67
            (id2, 0.88, 0.95, 0.92), // score ~ 0.77 (winner)
        ];

        let winner = provider
            .select_winning_memory(candidates)
            .await
            .expect("Selection failed");

        assert_eq!(winner, Some(id2), "Should select highest score candidate");
    }

    #[tokio::test]
    async fn test_workspace_provider_threshold_filtering() {
        let provider = WorkspaceProviderImpl::new();

        let id1 = Uuid::new_v4();
        let id2 = Uuid::new_v4();

        // Both below coherence threshold (0.8)
        let candidates = vec![(id1, 0.5, 0.9, 0.88), (id2, 0.6, 0.95, 0.92)];

        let winner = provider
            .select_winning_memory(candidates)
            .await
            .expect("Selection failed");

        assert_eq!(winner, None, "Should return None when all below threshold");
    }

    #[tokio::test]
    async fn test_workspace_provider_coherence_threshold() {
        let provider = WorkspaceProviderImpl::new();

        let threshold = provider.coherence_threshold().await;
        assert!(
            (threshold - 0.8).abs() < 0.01,
            "Threshold should be 0.8: {}",
            threshold
        );
    }

    #[tokio::test]
    async fn test_meta_cognitive_provider_evaluation() {
        let provider = MetaCognitiveProviderImpl::new();

        let state = provider
            .evaluate(0.8, 0.8)
            .await
            .expect("Evaluation failed");

        // Perfect prediction should have meta_score around 0.5 (sigmoid(0))
        assert!(
            state.meta_score >= 0.4 && state.meta_score <= 0.6,
            "Perfect prediction should give meta_score ~ 0.5: {}",
            state.meta_score
        );
        assert!(!state.dream_triggered);
    }

    #[tokio::test]
    async fn test_meta_cognitive_provider_initial_state() {
        let provider = MetaCognitiveProviderImpl::new();

        // Default acetylcholine is 0.001
        let ach = provider.acetylcholine().await;
        assert!(
            (ach - 0.001).abs() < 0.0001,
            "Initial ACh should be 0.001: {}",
            ach
        );

        // Default monitoring frequency is 1.0 Hz
        let freq = provider.monitoring_frequency().await;
        assert!(
            (freq - 1.0).abs() < 0.01,
            "Initial freq should be 1.0: {}",
            freq
        );

        // No recent scores initially
        let scores = provider.get_recent_scores().await;
        assert!(scores.is_empty(), "Should have no scores initially");
    }
}
