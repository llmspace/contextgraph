//! Global Workspace Theory (GWT) - Workspace Coordination
//!
//! Implements workspace coordination through per-space clustering and
//! Winner-Take-All workspace selection, as specified in Constitution v6.0.0.
//!
//! ## Architecture
//!
//! The GWT system consists of:
//!
//! 1. **Per-Space Clustering Coordination**: Topic-based coherence
//!    - Order parameter r measures clustering coherence level
//!    - Thresholds: r >= 0.8 (STABLE state), r < 0.5 (FRAGMENTED state)
//!
//! 2. **Global Workspace**: Winner-Take-All memory selection
//!    - Selects highest-scoring memory
//!    - Broadcasts to all subsystems
//!    - Enables unified perception
//!
//! 3. **State Machine**: Coherence state transitions
//!    - DORMANT -> FRAGMENTED -> EMERGING -> STABLE -> HYPERSYNC
//!    - Temporal dynamics based on coherence
//!
//! 4. **Meta-Cognitive Loop**: Self-correction
//!    - MetaScore = sigmoid(2*(L_predicted - L_actual))
//!    - Triggers Acetylcholine increase on low scores
//!    - Introspective dreams for error correction
//!
//! 5. **Workspace Events**: State transitions and signals
//!    - memory_enters_workspace: Dopamine reward
//!    - memory_exits_workspace: Dream replay logging
//!    - workspace_conflict: Multi-memory critique
//!    - workspace_empty: Epistemic action trigger

// Submodules
pub mod listeners;
pub mod meta_cognitive;
pub mod meta_learning_trait;
pub mod session_snapshot;
pub mod state_machine;
mod system;
pub mod workspace;

#[cfg(test)]
mod tests;

// Re-export from listeners
pub use listeners::{
    DreamEventListener, MetaCognitiveEventListener, NeuromodulationEventListener,
    WORKSPACE_EMPTY_THRESHOLD_MS,
};

// Re-export from meta_cognitive
pub use meta_cognitive::{MetaCognitiveLoop, MetaCognitiveState};

// Re-export from meta_learning_trait - TASK-METAUTL-P0-006
pub use meta_learning_trait::{
    EnhancedMetaCognitiveState, LambdaValues, MetaCallbackStatus, MetaDomain, MetaLambdaAdjustment,
    MetaLearningCallback, NoOpMetaLearningCallback,
};

// Re-export from state_machine
pub use state_machine::{
    CoherenceState, StateMachineManager, StateTransition, TransitionAnalysis,
};

// Re-export from workspace
pub use workspace::{
    GlobalWorkspace, WorkspaceCandidate, WorkspaceEvent, WorkspaceEventBroadcaster,
    WorkspaceEventListener, DA_INHIBITION_FACTOR,
};

// Re-export from system - the main GwtSystem orchestrator
pub use system::GwtSystem;

// Re-export TriggerManager from dream module for external use
pub use crate::dream::TriggerManager;

// Re-export from session_snapshot - TASK-SESSION-04
pub use session_snapshot::{
    store_in_cache, SessionCache, SessionManager, SessionSnapshot, MAX_TRAJECTORY_SIZE,
    NUM_EMBEDDERS,
};

