//! Workspace Event Listeners
//!
//! Implements listeners for workspace events that wire to subsystems:
//! - DreamEventListener: Queues exiting memories for dream replay
//! - NeuromodulationEventListener: Boosts dopamine on memory entry
//! - MetaCognitiveEventListener: Triggers epistemic action on workspace empty

mod dream;
mod meta_cognitive;
mod neuromod;

pub use dream::{DreamConsolidationCallback, DreamEventListener};
pub use meta_cognitive::{MetaCognitiveEventListener, WORKSPACE_EMPTY_THRESHOLD_MS};
pub use neuromod::NeuromodulationEventListener;

#[cfg(test)]
mod tests;
