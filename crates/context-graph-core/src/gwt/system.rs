//! GwtSystem - Global Workspace Theory system orchestrating workspace coordination
//!
//! This module contains the main GwtSystem struct that coordinates all
//! GWT components including per-space clustering coordination, workspace selection,
//! and event broadcasting.

use std::sync::atomic::AtomicBool;
use std::sync::Arc;
use tokio::sync::RwLock;
use uuid::Uuid;

// Import NeuromodulationManager for listener wiring
use crate::neuromod::NeuromodulationManager;

use super::{
    DreamEventListener, GlobalWorkspace, MetaCognitiveEventListener, MetaCognitiveLoop,
    NeuromodulationEventListener, StateMachineManager, WorkspaceEventBroadcaster,
};

/// Global Workspace Theory system orchestrating workspace coordination
#[derive(Debug)]
pub struct GwtSystem {
    /// Global workspace for winner-take-all selection
    pub workspace: Arc<RwLock<GlobalWorkspace>>,

    /// Coherence state machine
    pub state_machine: Arc<RwLock<StateMachineManager>>,

    /// Meta-cognitive feedback loop
    pub meta_cognitive: Arc<RwLock<MetaCognitiveLoop>>,

    /// Workspace event broadcaster
    pub event_broadcaster: Arc<WorkspaceEventBroadcaster>,

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

    /// TriggerManager for dream triggering
    ///
    /// Shared with DreamEventListener for dream scheduling.
    /// DreamScheduler also reads this to check trigger state.
    pub trigger_manager: Arc<parking_lot::Mutex<crate::dream::TriggerManager>>,
}

impl GwtSystem {
    /// Create a new GWT workspace coordination system
    ///
    /// Initializes all GWT components for workspace coordination.
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

        // Create TriggerManager for dream triggering
        let trigger_manager =
            Arc::new(parking_lot::Mutex::new(crate::dream::TriggerManager::new()));

        // Create event broadcaster
        let event_broadcaster = Arc::new(WorkspaceEventBroadcaster::new());

        // Create and register listeners
        let dream_consolidation_callback: crate::gwt::listeners::DreamConsolidationCallback =
            Arc::new(|reason| {
                // Log dream trigger for now - production may want to integrate with dream orchestrator
                tracing::info!("Dream consolidation triggered: {:?}", reason);
            });
        let dream_listener = DreamEventListener::new(
            Arc::clone(&dream_queue),
            Arc::clone(&trigger_manager),
            dream_consolidation_callback,
        );
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
            workspace: Arc::new(RwLock::new(GlobalWorkspace::new())),
            state_machine: Arc::new(RwLock::new(StateMachineManager::new())),
            meta_cognitive,
            event_broadcaster,
            neuromod_manager,
            dream_queue,
            epistemic_action_triggered,
            trigger_manager,
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
        self.epistemic_action_triggered
            .store(false, Ordering::SeqCst);
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

    /// Select winning memory for workspace broadcast
    pub async fn select_workspace_memory(
        &self,
        candidates: Vec<(Uuid, f32, f32, f32)>, // (id, r, importance, alignment)
    ) -> crate::CoreResult<Option<Uuid>> {
        let mut workspace = self.workspace.write().await;
        workspace.select_winning_memory(candidates).await
    }

    /// Get reference to the TriggerManager for external use.
    ///
    /// Returns Arc clone for DreamScheduler integration.
    pub fn trigger_manager(&self) -> Arc<parking_lot::Mutex<crate::dream::TriggerManager>> {
        Arc::clone(&self.trigger_manager)
    }
}
