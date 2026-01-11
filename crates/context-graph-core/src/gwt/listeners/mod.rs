//! Workspace Event Listeners
//!
//! Implements listeners for workspace events that wire to subsystems:
//! - DreamEventListener: Queues exiting memories for dream replay
//! - NeuromodulationEventListener: Boosts dopamine on memory entry
//! - MetaCognitiveEventListener: Triggers epistemic action on workspace empty
//!
//! ## Constitution Reference
//!
//! From constitution.yaml:
//! - neuromod.Dopamine.trigger: "memory_enters_workspace" (lines 162-170)
//! - gwt.global_workspace step 6: "Inhibit: losing candidates receive dopamine reduction"
//! - gwt.workspace_events: memory_exits → dream replay, workspace_empty → epistemic action

mod dream;
mod meta_cognitive;
mod neuromod;

pub use dream::DreamEventListener;
pub use meta_cognitive::MetaCognitiveEventListener;
pub use neuromod::NeuromodulationEventListener;

#[cfg(test)]
mod tests;
