//! IdentityContinuityMonitor - Continuous IC tracking wrapper
//!
//! Wraps PurposeVectorHistory to provide real-time identity continuity
//! monitoring and status classification.

use serde::{Deserialize, Serialize};

use super::cosine::cosine_similarity_13d;
use super::identity_continuity::IdentityContinuity;
use super::purpose_vector_history::{PurposeVectorHistory, PurposeVectorHistoryProvider};
use super::types::{IdentityStatus, IC_CRISIS_THRESHOLD};

/// Identity Continuity Monitor - Continuous IC tracking wrapper
///
/// Wraps `PurposeVectorHistory` to provide real-time identity continuity
/// monitoring and status classification.
///
/// # Constitution Reference
/// From constitution.yaml lines 365-392:
/// - IC = cos(PV_t, PV_{t-1}) x r(t)
/// - Thresholds: healthy>0.9, warning<0.7, dream<0.5
/// - self_ego_node.identity_trajectory: max 1000 snapshots
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct IdentityContinuityMonitor {
    /// Purpose vector history buffer (delegates to PurposeVectorHistory)
    history: PurposeVectorHistory,
    /// Cached last computation result (None until first compute_continuity call)
    last_result: Option<IdentityContinuity>,
    /// Configurable crisis threshold (default: IC_CRISIS_THRESHOLD = 0.7)
    crisis_threshold: f32,
}

impl IdentityContinuityMonitor {
    /// Create new monitor with default configuration.
    ///
    /// Defaults:
    /// - history capacity: MAX_PV_HISTORY_SIZE (1000)
    /// - crisis_threshold: IC_CRISIS_THRESHOLD (0.7)
    pub fn new() -> Self {
        Self {
            history: PurposeVectorHistory::new(),
            last_result: None,
            crisis_threshold: IC_CRISIS_THRESHOLD,
        }
    }

    /// Create monitor with custom crisis threshold.
    ///
    /// # Arguments
    /// * `threshold` - Custom crisis threshold (clamped to [0, 1])
    pub fn with_threshold(threshold: f32) -> Self {
        Self {
            history: PurposeVectorHistory::new(),
            last_result: None,
            crisis_threshold: threshold.clamp(0.0, 1.0),
        }
    }

    /// Create monitor with custom history capacity.
    ///
    /// # Arguments
    /// * `capacity` - Maximum history entries (0 = unlimited)
    pub fn with_capacity(capacity: usize) -> Self {
        Self {
            history: PurposeVectorHistory::with_max_size(capacity),
            last_result: None,
            crisis_threshold: IC_CRISIS_THRESHOLD,
        }
    }

    /// Compute identity continuity from new purpose vector and Kuramoto r.
    ///
    /// # Algorithm
    /// 1. Push new PV to history, get previous PV
    /// 2. If first vector: return IdentityContinuity::first_vector()
    /// 3. Compute cos(PV_t, PV_{t-1}) using cosine_similarity_13d
    /// 4. Create IdentityContinuity::new(cosine, kuramoto_r)
    /// 5. Cache and return result
    ///
    /// # Arguments
    /// * `purpose_vector` - Current 13D purpose alignment vector (PV_t)
    /// * `kuramoto_r` - Current Kuramoto order parameter r(t) in [0, 1]
    /// * `context` - Description for history snapshot
    ///
    /// # Returns
    /// * `IdentityContinuity` with computed IC and status
    pub fn compute_continuity(
        &mut self,
        purpose_vector: &[f32; 13],
        kuramoto_r: f32,
        context: impl Into<String>,
    ) -> IdentityContinuity {
        // Push current PV and get previous (if any)
        let previous = self.history.push(*purpose_vector, context);

        // Compute result based on whether this is first vector
        let result = match previous {
            None => {
                // First vector: per EC-IDENTITY-01, default to healthy
                IdentityContinuity::first_vector()
            }
            Some(prev_pv) => {
                // Compute cosine similarity between consecutive PVs
                let cosine = cosine_similarity_13d(purpose_vector, &prev_pv);

                // Create IdentityContinuity with IC = cos x r
                IdentityContinuity::new(cosine, kuramoto_r)
            }
        };

        // Cache result for subsequent getters
        self.last_result = Some(result.clone());

        result
    }

    /// Get the last computed IdentityContinuity result.
    #[inline]
    pub fn last_result(&self) -> Option<&IdentityContinuity> {
        self.last_result.as_ref()
    }

    /// Get current identity coherence value (IC).
    #[inline]
    pub fn identity_coherence(&self) -> Option<f32> {
        self.last_result.as_ref().map(|r| r.identity_coherence)
    }

    /// Get current identity status classification.
    #[inline]
    pub fn current_status(&self) -> Option<IdentityStatus> {
        self.last_result.as_ref().map(|r| r.status)
    }

    /// Check if identity is in crisis (IC < crisis_threshold).
    #[inline]
    pub fn is_in_crisis(&self) -> bool {
        self.last_result
            .as_ref()
            .map(|r| r.identity_coherence < self.crisis_threshold)
            .unwrap_or(false)
    }

    /// Get the number of snapshots in history.
    #[inline]
    pub fn history_len(&self) -> usize {
        self.history.len()
    }

    /// Get the configured crisis threshold.
    #[inline]
    pub fn crisis_threshold(&self) -> f32 {
        self.crisis_threshold
    }

    /// Get read-only access to underlying history.
    pub fn history(&self) -> &PurposeVectorHistory {
        &self.history
    }

    /// Check if history is empty (no vectors recorded).
    #[inline]
    pub fn is_empty(&self) -> bool {
        self.history.is_empty()
    }

    /// Check if this is the first vector (exactly one entry).
    #[inline]
    pub fn is_first_vector(&self) -> bool {
        self.history.is_first_vector()
    }
}

impl Default for IdentityContinuityMonitor {
    fn default() -> Self {
        Self::new()
    }
}
