//! IdentityContinuity - Identity coherence tracking
//!
//! Tracks identity continuity over time per constitution.yaml lines 365-392.

use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};

use crate::error::CoreResult;
use super::types::IdentityStatus;

/// Tracks identity continuity over time
///
/// # Constitution Reference
/// From constitution.yaml lines 365-392:
/// - identity_continuity: "IC = cos(PV_t, PV_{t-1}) x r(t)"
/// - Thresholds: healthy>0.9, warning<0.7, dream<0.5
///
/// # Persistence (TASK-GWT-P1-001)
/// Serializable for diagnostic/recovery purposes.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct IdentityContinuity {
    /// IC = cos(PV_t, PV_{t-1}) x r(t), clamped to [0, 1]
    pub identity_coherence: f32,
    /// Cosine similarity between consecutive purpose vectors
    pub recent_continuity: f32,
    /// Order parameter r from Kuramoto sync
    pub kuramoto_order_parameter: f32,
    /// Status classification based on IC thresholds
    pub status: IdentityStatus,
    /// Timestamp of computation
    pub computed_at: DateTime<Utc>,
}

impl IdentityContinuity {
    /// Create a new result with computed values
    ///
    /// # Arguments
    /// * `purpose_continuity` - cos(PV_t, PV_{t-1})
    /// * `kuramoto_r` - Kuramoto order parameter r(t)
    ///
    /// # Returns
    /// Result with IC = purpose_continuity * kuramoto_r, clamped to [0, 1]
    pub fn new(purpose_continuity: f32, kuramoto_r: f32) -> Self {
        // Clamp inputs to valid ranges
        let cos_clamped = purpose_continuity.clamp(-1.0, 1.0);
        let r_clamped = kuramoto_r.clamp(0.0, 1.0);

        // Compute IC = cos * r, clamp negative to 0
        let ic = (cos_clamped * r_clamped).clamp(0.0, 1.0);

        // Determine status from IC
        let status = Self::compute_status_from_coherence(ic);

        Self {
            identity_coherence: ic,
            recent_continuity: cos_clamped,
            kuramoto_order_parameter: r_clamped,
            status,
            computed_at: Utc::now(),
        }
    }

    /// Create new IdentityContinuity with default initial state
    ///
    /// Starting with identity_coherence=0.0 means status=Critical (IC < 0.5)
    /// per constitution.yaml lines 387-392
    pub fn default_initial() -> Self {
        let identity_coherence = 0.0;
        Self {
            identity_coherence,
            recent_continuity: 1.0,
            kuramoto_order_parameter: 0.0,
            status: Self::compute_status_from_coherence(identity_coherence),
            computed_at: Utc::now(),
        }
    }

    /// Create result for first purpose vector (no previous)
    ///
    /// Returns IC = 1.0, Status = Healthy
    /// Per EC-IDENTITY-01: First purpose vector defaults to healthy
    pub fn first_vector() -> Self {
        Self {
            identity_coherence: 1.0,
            recent_continuity: 1.0,
            kuramoto_order_parameter: 1.0,
            status: IdentityStatus::Healthy,
            computed_at: Utc::now(),
        }
    }

    /// Check if identity is in crisis (IC < 0.7)
    ///
    /// # Constitution Reference
    /// From constitution.yaml line 369:
    /// - warning<0.7 threshold indicates identity drift
    #[inline]
    pub fn is_in_crisis(&self) -> bool {
        self.identity_coherence < 0.7
    }

    /// Check if identity is critical (IC < 0.5)
    ///
    /// # Constitution Reference
    /// From constitution.yaml line 369:
    /// - dream<0.5 threshold triggers introspective dream
    #[inline]
    pub fn is_critical(&self) -> bool {
        self.identity_coherence < 0.5
    }

    /// Compute status per constitution.yaml lines 387-392:
    /// - Healthy: IC > 0.9
    /// - Warning: 0.7 <= IC <= 0.9
    /// - Degraded: 0.5 <= IC < 0.7
    /// - Critical: IC < 0.5 (triggers dream consolidation)
    pub(crate) fn compute_status_from_coherence(coherence: f32) -> IdentityStatus {
        match coherence {
            ic if ic > 0.9 => IdentityStatus::Healthy,
            ic if ic >= 0.7 => IdentityStatus::Warning,
            ic if ic >= 0.5 => IdentityStatus::Degraded,
            _ => IdentityStatus::Critical,
        }
    }

    /// Update identity coherence: IC = cos(PV_t, PV_{t-1}) x r(t)
    pub fn update(&mut self, pv_cosine: f32, kuramoto_r: f32) -> CoreResult<IdentityStatus> {
        self.recent_continuity = pv_cosine.clamp(-1.0, 1.0);
        self.kuramoto_order_parameter = kuramoto_r.clamp(0.0, 1.0);

        // Identity coherence = cosine x r
        self.identity_coherence = (pv_cosine * kuramoto_r).clamp(0.0, 1.0);

        // Determine status using canonical computation
        self.status = Self::compute_status_from_coherence(self.identity_coherence);

        // Update timestamp
        self.computed_at = Utc::now();

        Ok(self.status)
    }
}

impl Default for IdentityContinuity {
    fn default() -> Self {
        Self::default_initial()
    }
}
