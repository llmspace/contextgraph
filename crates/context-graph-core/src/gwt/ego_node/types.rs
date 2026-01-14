//! Type definitions for SELF_EGO_NODE
//!
//! Contains constants, basic types, and enums used across the ego_node module.

use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use std::time::Duration;

/// Maximum purpose vector history size per constitution
/// Reference: constitution.yaml line 390 (identity_trajectory: 1000)
pub const MAX_PV_HISTORY_SIZE: usize = 1000;

/// IC threshold for CRITICAL status (IC < this = CRITICAL, triggers dream).
///
/// Per IDENTITY-002: IC < 0.5 is CRITICAL and MUST trigger dream consolidation.
///
/// # Constitution Reference
///
/// - IDENTITY-002: IC < 0.5 = CRITICAL
/// - IDENTITY-007: IC < 0.5 -> auto-trigger dream
/// - GWT-003: IC < 0.5 -> dream consolidation
/// - AP-26: No silent failures on IC crisis
pub const IC_CRITICAL_THRESHOLD: f32 = 0.5;

/// IC threshold for WARNING status.
///
/// Per IDENTITY-002: IC in [0.7, 0.9) is WARNING.
pub const IC_WARNING_THRESHOLD: f32 = 0.7;

/// IC threshold for HEALTHY status.
///
/// Per IDENTITY-002: IC >= 0.9 is HEALTHY.
pub const IC_HEALTHY_THRESHOLD: f32 = 0.9;

/// Cooldown between crisis events to prevent spam.
///
/// Per TASK-IDENTITY-P0-004: Crisis Detection
pub const CRISIS_EVENT_COOLDOWN: Duration = Duration::from_millis(1000);

// DEPRECATED: Alias for backwards compatibility during migration
/// Default crisis threshold per constitution.yaml line 369.
/// IC < 0.7 indicates identity drift (warning/degraded state).
///
/// **DEPRECATED**: Use `IC_CRITICAL_THRESHOLD` for critical (< 0.5) or
/// `IC_WARNING_THRESHOLD` for warning (< 0.7) checks.
#[deprecated(
    since = "5.0.0",
    note = "Use IC_CRITICAL_THRESHOLD (0.5) for critical checks or IC_WARNING_THRESHOLD (0.7) for degraded checks"
)]
pub const IC_CRISIS_THRESHOLD: f32 = IC_WARNING_THRESHOLD;

/// Epsilon for numerical stability in magnitude comparisons
/// Prevents division by zero in cosine similarity calculation
pub(crate) const COSINE_EPSILON: f32 = 1e-8;

/// Snapshot of purpose vector at a point in time
///
/// # Persistence (TASK-GWT-P1-001)
///
/// Serializable component of SelfEgoNode's identity_trajectory.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PurposeSnapshot {
    /// Purpose vector at this moment
    pub vector: [f32; 13],
    /// Timestamp of snapshot
    pub timestamp: DateTime<Utc>,
    /// Context (brief description of system state)
    pub context: String,
}

/// Identity status enum for SELF_EGO_NODE state tracking.
///
/// # Persistence (TASK-GWT-P1-001)
///
/// Serializable component of IdentityContinuity.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum IdentityStatus {
    /// IC > 0.9: Healthy identity continuity
    Healthy,
    /// 0.7 <= IC <= 0.9: Warning state, monitor closely
    Warning,
    /// IC < 0.7: Degraded identity, may need intervention
    Degraded,
    /// IC < 0.5: Critical, trigger introspective dream
    Critical,
}

/// Result from a self-awareness cycle
#[derive(Debug, Clone)]
pub struct SelfReflectionResult {
    /// Alignment between action and purpose
    pub alignment: f32,
    /// Whether self-reflection should be triggered
    pub needs_reflection: bool,
    /// Current identity status
    pub identity_status: IdentityStatus,
    /// Current identity coherence value
    pub identity_coherence: f32,
}
