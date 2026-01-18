//! Global Workspace Selection - Winner-Take-All Algorithm
//!
//! Implements conscious memory selection via winner-take-all (WTA) competition
//! as specified in Constitution v4.0.0 Section gwt.global_workspace (lines 352-369).
//!
//! ## Algorithm
//!
//! 1. Compute coherence order parameter r for all candidate memories
//! 2. Filter: candidates where r >= coherence_threshold (0.8)
//! 3. Rank: score = r * importance * alignment
//! 4. Select: top-1 becomes active_memory
//! 5. Broadcast: active_memory visible to all subsystems (100ms window)
//! 6. Inhibit: losing candidates receive dopamine reduction
//!
//! ## Module Structure
//!
//! - `candidate` - WorkspaceCandidate struct for competing memories
//! - `global` - GlobalWorkspace for WTA selection
//! - `events` - Event broadcasting system for workspace state changes

mod candidate;
mod events;
mod global;

// Re-export all public types for backwards compatibility
pub use candidate::WorkspaceCandidate;
pub use events::{WorkspaceEvent, WorkspaceEventBroadcaster, WorkspaceEventListener};
pub use global::GlobalWorkspace;

/// Dopamine inhibition factor for WTA losers
/// Per constitution gwt.global_workspace step 6: "Inhibit: losing candidates receive dopamine reduction"
pub const DA_INHIBITION_FACTOR: f32 = 0.1;
