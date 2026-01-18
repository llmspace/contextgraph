//! Autonomous-specific error codes for MCP handlers.
//!
//! Error codes in the range -32110 to -32119.

/// Autonomous-specific error codes (-32110 to -32119)
#[allow(dead_code)]
pub mod autonomous_error_codes {
    /// Bootstrap service failed
    pub const BOOTSTRAP_ERROR: i32 = -32110;
    /// Drift detector failed
    pub const DRIFT_DETECTOR_ERROR: i32 = -32111;
    /// Drift corrector failed
    pub const DRIFT_CORRECTOR_ERROR: i32 = -32112;
    /// Pruning service failed
    pub const PRUNING_ERROR: i32 = -32113;
    /// Consolidation service failed
    pub const CONSOLIDATION_ERROR: i32 = -32114;
    /// Sub-goal discovery failed
    pub const SUBGOAL_DISCOVERY_ERROR: i32 = -32115;
    /// Autonomous status aggregation failed
    pub const STATUS_AGGREGATION_ERROR: i32 = -32116;
}
