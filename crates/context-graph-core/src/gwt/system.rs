//! GwtSystem - Global Workspace Theory system orchestrating consciousness
//!
//! This module contains the main GwtSystem struct that coordinates all
//! GWT components including Kuramoto synchronization, workspace selection,
//! self-awareness loop, and event broadcasting.
//!
//! # Module Organization
//!
//! The GwtSystem implementation is split across multiple files:
//! - `system.rs` (this file): Struct definition, constructor, and basic accessors
//! - `system_kuramoto.rs`: Kuramoto oscillator methods and consciousness updates
//! - `system_awareness.rs`: Self-awareness loop and identity crisis handling

use std::sync::atomic::AtomicBool;
use std::sync::Arc;
use tokio::sync::RwLock;
use uuid::Uuid;

// Import KuramotoNetwork and constants from layers module
use crate::layers::{KuramotoNetwork, KURAMOTO_K, KURAMOTO_N};
// Import NeuromodulationManager for listener wiring
use crate::neuromod::NeuromodulationManager;

use super::{
    ConsciousnessCalculator, DreamEventListener, GlobalWorkspace,
    MetaCognitiveEventListener, MetaCognitiveLoop, NeuromodulationEventListener,
    SelfAwarenessLoop, SelfEgoNode, StateMachineManager,
    WorkspaceEventBroadcaster,
};

/// Global Workspace Theory system orchestrating consciousness
#[derive(Debug)]
pub struct GwtSystem {
    /// Consciousness calculator (C = I×R×D)
    pub consciousness_calc: Arc<ConsciousnessCalculator>,

    /// Global workspace for winner-take-all selection
    pub workspace: Arc<RwLock<GlobalWorkspace>>,

    /// System identity node
    pub self_ego_node: Arc<RwLock<SelfEgoNode>>,

    /// Consciousness state machine
    pub state_machine: Arc<RwLock<StateMachineManager>>,

    /// Meta-cognitive feedback loop
    pub meta_cognitive: Arc<RwLock<MetaCognitiveLoop>>,

    /// Workspace event broadcaster
    pub event_broadcaster: Arc<WorkspaceEventBroadcaster>,

    /// Kuramoto oscillator network for phase synchronization (I(t) computation)
    ///
    /// Uses 8 oscillators from layers::coherence for layer-level sync.
    /// The order parameter r measures synchronization level in [0, 1].
    pub kuramoto: Arc<RwLock<KuramotoNetwork>>,

    /// Self-awareness loop for identity continuity monitoring
    ///
    /// From constitution.yaml lines 365-392:
    /// - loop: "Retrieve→A(action,PV)→if<0.55 self_reflect→update fingerprint→store evolution"
    /// - identity_continuity: "IC = cos(PV_t, PV_{t-1}) × r(t); healthy>0.9, warning<0.7, dream<0.5"
    pub self_awareness_loop: Arc<RwLock<SelfAwarenessLoop>>,

    /// Neuromodulation manager for dopamine/serotonin/NE control
    ///
    /// Wired to workspace events for dopamine modulation on memory entry.
    pub neuromod_manager: Arc<RwLock<NeuromodulationManager>>,

    /// Queue of memories that exited workspace, pending dream replay
    ///
    /// DreamController consumes this queue during dream cycles.
    pub dream_queue: Arc<RwLock<Vec<Uuid>>>,

    /// Flag set when workspace is empty, triggering epistemic action
    ///
    /// MetaCognitiveLoop uses this to trigger exploratory behavior.
    pub epistemic_action_triggered: Arc<AtomicBool>,
}

impl GwtSystem {
    /// Create a new GWT consciousness system
    ///
    /// Initializes all GWT components including the Kuramoto oscillator network
    /// for phase synchronization and consciousness computation.
    ///
    /// # Listener Wiring
    ///
    /// The following listeners are automatically registered:
    /// - `DreamEventListener`: Queues exiting memories for dream replay
    /// - `NeuromodulationEventListener`: Boosts dopamine on memory entry
    /// - `MetaCognitiveEventListener`: Triggers epistemic action on workspace empty
    pub async fn new() -> crate::CoreResult<Self> {
        // Create shared state for listeners
        let neuromod_manager = Arc::new(RwLock::new(NeuromodulationManager::new()));
        let meta_cognitive = Arc::new(RwLock::new(MetaCognitiveLoop::new()));
        let dream_queue: Arc<RwLock<Vec<Uuid>>> = Arc::new(RwLock::new(Vec::new()));
        let epistemic_action_triggered = Arc::new(AtomicBool::new(false));

        // Create event broadcaster
        let event_broadcaster = Arc::new(WorkspaceEventBroadcaster::new());

        // Create and register listeners
        let dream_listener = DreamEventListener::new(Arc::clone(&dream_queue));
        let neuromod_listener = NeuromodulationEventListener::new(Arc::clone(&neuromod_manager));
        let meta_listener = MetaCognitiveEventListener::new(
            Arc::clone(&meta_cognitive),
            Arc::clone(&epistemic_action_triggered),
        );

        event_broadcaster
            .register_listener(Box::new(dream_listener))
            .await;
        event_broadcaster
            .register_listener(Box::new(neuromod_listener))
            .await;
        event_broadcaster
            .register_listener(Box::new(meta_listener))
            .await;

        tracing::info!(
            "GwtSystem initialized with {} event listeners",
            event_broadcaster.listener_count().await
        );

        Ok(Self {
            consciousness_calc: Arc::new(ConsciousnessCalculator::new()),
            workspace: Arc::new(RwLock::new(GlobalWorkspace::new())),
            self_ego_node: Arc::new(RwLock::new(SelfEgoNode::new())),
            state_machine: Arc::new(RwLock::new(StateMachineManager::new())),
            meta_cognitive,
            event_broadcaster,
            kuramoto: Arc::new(RwLock::new(KuramotoNetwork::new(KURAMOTO_N, KURAMOTO_K))),
            self_awareness_loop: Arc::new(RwLock::new(SelfAwarenessLoop::new())),
            neuromod_manager,
            dream_queue,
            epistemic_action_triggered,
        })
    }

    /// Check if epistemic action has been triggered
    pub fn is_epistemic_action_triggered(&self) -> bool {
        use std::sync::atomic::Ordering;
        self.epistemic_action_triggered.load(Ordering::SeqCst)
    }

    /// Reset the epistemic action flag
    pub fn reset_epistemic_action(&self) {
        use std::sync::atomic::Ordering;
        self.epistemic_action_triggered.store(false, Ordering::SeqCst);
    }

    /// Get the number of memories pending dream replay
    pub async fn dream_queue_len(&self) -> usize {
        let queue = self.dream_queue.read().await;
        queue.len()
    }

    /// Take all memories from the dream queue (for DreamController)
    pub async fn drain_dream_queue(&self) -> Vec<Uuid> {
        let mut queue = self.dream_queue.write().await;
        std::mem::take(&mut *queue)
    }

    /// Get reference to the Kuramoto network
    ///
    /// Returns an Arc clone for concurrent access to the oscillator network.
    pub fn kuramoto(&self) -> Arc<RwLock<KuramotoNetwork>> {
        Arc::clone(&self.kuramoto)
    }

    /// Select winning memory for workspace broadcast
    pub async fn select_workspace_memory(
        &self,
        candidates: Vec<(Uuid, f32, f32, f32)>, // (id, r, importance, alignment)
    ) -> crate::CoreResult<Option<Uuid>> {
        let mut workspace = self.workspace.write().await;
        workspace.select_winning_memory(candidates).await
    }
}
