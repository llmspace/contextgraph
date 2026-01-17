//! SelfAwarenessLoop - Self-awareness loop for identity continuity
//!
//! Implements the self-awareness cycle from constitution.yaml lines 365-392.

use super::identity_continuity::IdentityContinuity;
use super::self_ego_node::SelfEgoNode;
use super::types::{IdentityStatus, SelfReflectionResult};
use crate::error::CoreResult;

/// Self-Awareness Loop for identity continuity
#[derive(Debug)]
pub struct SelfAwarenessLoop {
    /// Identity continuity tracking
    continuity: IdentityContinuity,
    /// Action-to-purpose alignment threshold
    alignment_threshold: f32,
}

impl SelfAwarenessLoop {
    /// Create a new self-awareness loop
    pub fn new() -> Self {
        Self {
            continuity: IdentityContinuity::default_initial(),
            alignment_threshold: 0.55,
        }
    }

    /// Get the current identity coherence value
    ///
    /// Returns the IC value computed as: IC = cos(PV_t, PV_{t-1}) x r(t)
    /// Per constitution.yaml lines 387-392
    pub fn identity_coherence(&self) -> f32 {
        self.continuity.identity_coherence
    }

    /// Get the current identity status
    pub fn identity_status(&self) -> IdentityStatus {
        self.continuity.status
    }

    /// Execute self-awareness loop for a single cycle
    ///
    /// # Algorithm
    /// 1. Retrieve current SELF_EGO_NODE purpose vector
    /// 2. Compute alignment with current action
    /// 3. If alignment < 0.55: trigger self-reflection
    /// 4. Update fingerprint with action outcome
    /// 5. Store to purpose_evolution (temporal trajectory)
    pub async fn cycle(
        &mut self,
        ego_node: &mut SelfEgoNode,
        action_embedding: &[f32; 13],
        kuramoto_r: f32,
    ) -> CoreResult<SelfReflectionResult> {
        // Compute cosine similarity between action and current purpose
        let alignment = self.cosine_similarity(&ego_node.purpose_vector, action_embedding);

        // Check if reflection is needed
        let needs_reflection = alignment < self.alignment_threshold;

        // Update identity continuity
        if !ego_node.identity_trajectory.is_empty() {
            let prev_pv = ego_node
                .get_latest_snapshot()
                .map(|s| s.vector)
                .unwrap_or(ego_node.purpose_vector);

            let pv_cosine = self.cosine_similarity(&prev_pv, &ego_node.purpose_vector);
            let status = self.continuity.update(pv_cosine, kuramoto_r)?;

            // Check for critical identity drift
            if status == IdentityStatus::Critical {
                // Trigger introspective dream
                ego_node.record_purpose_snapshot("Critical identity drift - dream triggered")?;
            }
        }

        // Record snapshot of current state
        ego_node.record_purpose_snapshot("Self-awareness cycle")?;

        Ok(SelfReflectionResult {
            alignment,
            needs_reflection,
            identity_status: self.continuity.status,
            identity_coherence: self.continuity.identity_coherence,
        })
    }

    /// Compute cosine similarity between two 13D vectors
    fn cosine_similarity(&self, v1: &[f32; 13], v2: &[f32; 13]) -> f32 {
        let dot_product: f32 = v1.iter().zip(v2.iter()).map(|(a, b)| a * b).sum();
        let magnitude_v1: f32 = (v1.iter().map(|a| a * a).sum::<f32>()).sqrt();
        let magnitude_v2: f32 = (v2.iter().map(|a| a * a).sum::<f32>()).sqrt();

        if magnitude_v1 < 1e-6 || magnitude_v2 < 1e-6 {
            0.0
        } else {
            dot_product / (magnitude_v1 * magnitude_v2)
        }
    }
}

impl Default for SelfAwarenessLoop {
    fn default() -> Self {
        Self::new()
    }
}
