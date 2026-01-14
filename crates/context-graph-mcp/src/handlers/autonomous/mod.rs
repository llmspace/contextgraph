//! NORTH Autonomous System MCP Handlers
//!
//! TASK-AUTONOMOUS-MCP: MCP tool handlers for autonomous North Star management.
//! SPEC-AUTONOMOUS-001: Added 5 new tools (learner, health, execute_prune).
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
//! 8. `get_learner_state` - Get Meta-UTL learner state (NORTH-009)
//! 9. `observe_outcome` - Record learning outcome (NORTH-009)
//! 10. `execute_prune` - Execute pruning on candidates (NORTH-012)
//! 11. `get_health_status` - Get system-wide health (NORTH-020)
//! 12. `trigger_healing` - Trigger self-healing (NORTH-020)
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
//! - `maintenance`: Pruning, consolidation, and execute_prune handlers
//! - `discovery`: Sub-goal discovery handler
//! - `status`: Autonomous status handler
//! - `learner`: Meta-UTL learner state and outcome handlers (SPEC-AUTONOMOUS-001)
//! - `health`: System health and healing handlers (SPEC-AUTONOMOUS-001)

mod bootstrap;
mod discovery;
mod drift;
mod error_codes;
mod health;
mod learner;
mod maintenance;
mod params;
mod prediction_history;
mod status;

// TASK-004: Re-export PredictionHistory for use in Handlers struct
pub use prediction_history::{PredictionEntry, PredictionHistory};

#[cfg(test)]
mod tests;

// Re-export all parameter structs for backwards compatibility
#[allow(unused_imports)]
pub use params::{
    AutoBootstrapParams, DiscoverSubGoalsParams, ExecutePruneParams, GetAlignmentDriftParams,
    GetAutonomousStatusParams, GetHealthStatusParams, GetLearnerStateParams,
    GetPruningCandidatesParams, ObserveOutcomeParams, TriggerConsolidationParams,
    TriggerDriftCorrectionParams, TriggerHealingParams,
};

// Re-export error codes module for backwards compatibility
#[allow(unused_imports)]
pub use error_codes::autonomous_error_codes;
