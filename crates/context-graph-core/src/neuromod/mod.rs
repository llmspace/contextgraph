//! Neuromodulation System - Dynamic System Modulation
//!
//! Implements 4 neuromodulators that affect system behavior:
//! - Dopamine (DA): Controls Hopfield beta, triggered by workspace events
//! - Serotonin (5HT): Scales embedding space weights E1-E13
//! - Noradrenaline (NE): Controls attention temperature
//! - Acetylcholine (ACh): Controls UTL learning rate (integrated with GWT)
//!
//! ## Constitution Reference: neuromod (lines 162-206)
//!
//! ```yaml
//! neuromod:
//!   Dopamine:
//!     range: "[1, 5]"
//!     parameter: hopfield.beta
//!     trigger: memory_enters_workspace  # GWT event
//!   Serotonin:
//!     range: "[0, 1]"
//!     parameter: space_weights  # E1-E13 modulation
//!     effect: space_weight_scaling  # scales embedding space weights
//!   Noradrenaline:
//!     range: "[0.5, 2]"
//!     parameter: attention.temp
//!     trigger: threat_detection
//!   Acetylcholine:
//!     range: "[0.001, 0.002]"
//!     parameter: utl.lr
//!     trigger: meta_cognitive.dream
//! ```
//!
//! ## Architecture
//!
//! Each neuromodulator has:
//! 1. A defined range with baseline (center)
//! 2. Triggers that cause modulation
//! 3. Homeostatic decay toward baseline
//! 4. Effects on specific system parameters
//!
//! ## Usage
//!
//! ```ignore
//! use context_graph_core::neuromod::{NeuromodulationManager, ModulatorType};
//!
//! let mut manager = NeuromodulationManager::new();
//!
//! // Trigger dopamine on workspace entry
//! manager.on_workspace_entry();
//!
//! // Get current Hopfield beta
//! let beta = manager.get_hopfield_beta();
//!
//! // Apply homeostatic decay
//! manager.decay_all(delta_time);
//! ```

pub mod acetylcholine;
pub mod dopamine;
pub mod noradrenaline;
pub mod serotonin;
pub mod state;

// Re-exports for convenience
pub use acetylcholine::{AcetylcholineProvider, ACH_BASELINE, ACH_MAX};
pub use dopamine::{
    DopamineLevel, DopamineModulator, DA_BASELINE, DA_GOAL_SENSITIVITY, DA_MAX, DA_MIN,
};
pub use noradrenaline::{NoradrenalineLevel, NoradrenalineModulator, NE_BASELINE, NE_MAX, NE_MIN};
pub use serotonin::{
    SerotoninLevel, SerotoninModulator, NUM_EMBEDDING_SPACES, SEROTONIN_BASELINE, SEROTONIN_MAX,
    SEROTONIN_MIN,
};
pub use state::{
    cascade, CascadeReport, ModulatorType, NeuromodulationManager, NeuromodulationState,
};
