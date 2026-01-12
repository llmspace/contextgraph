//! NORTH Autonomous System MCP Handlers
//!
//! TASK-AUTONOMOUS-MCP: MCP tool handlers for autonomous North Star management.
//! NO BACKWARDS COMPATIBILITY - FAIL FAST WITH ROBUST LOGGING.
//!
//! ## Tools Implemented
//!
//! 1. `auto_bootstrap_north_star` - Bootstrap from existing North Star using BootstrapService
//! 2. `get_alignment_drift` - Get drift state and history using DriftDetector
//! 3. `trigger_drift_correction` - Manually trigger correction using DriftCorrector
//! 4. `get_pruning_candidates` - Get memories for potential pruning using PruningService
//! 5. `trigger_consolidation` - Trigger memory consolidation using ConsolidationService
//! 6. `discover_sub_goals` - Discover potential sub-goals using SubGoalDiscovery
//! 7. `get_autonomous_status` - Get comprehensive status from all services
//!
//! ## FAIL FAST Policy
//!
//! - NO MOCK DATA - all calls go to real services
//! - NO FALLBACKS - errors propagate with full context
//! - All errors include operation context for debugging
//!
//! ## Module Organization
//!
//! - `params`: Parameter structs and default value functions
//! - `error_codes`: Autonomous-specific error codes
//! - `bootstrap`: Bootstrap handler implementation
//! - `drift`: Drift detection and correction handlers
//! - `maintenance`: Pruning and consolidation handlers
//! - `discovery`: Sub-goal discovery handler
//! - `status`: Autonomous status handler

mod bootstrap;
mod discovery;
mod drift;
mod error_codes;
mod maintenance;
mod params;
mod status;

#[cfg(test)]
mod tests;

// Re-export all parameter structs for backwards compatibility
#[allow(unused_imports)]
pub use params::{
    AutoBootstrapParams, DiscoverSubGoalsParams, GetAlignmentDriftParams,
    GetAutonomousStatusParams, GetPruningCandidatesParams, TriggerConsolidationParams,
    TriggerDriftCorrectionParams,
};

// Re-export error codes module for backwards compatibility
#[allow(unused_imports)]
pub use error_codes::autonomous_error_codes;
