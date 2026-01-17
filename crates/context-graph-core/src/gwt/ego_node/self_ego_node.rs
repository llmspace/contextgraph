//! SelfEgoNode - System Identity Node
//!
//! Special memory node representing the system's identity.

use crate::error::CoreResult;
use crate::types::fingerprint::TeleologicalFingerprint;
use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use uuid::Uuid;

use super::types::PurposeSnapshot;

/// Special memory node representing the system's identity
///
/// # Persistence (TASK-GWT-P1-001)
///
/// This struct is serializable via Serde for RocksDB storage in CF_EGO_NODE.
/// Uses bincode for efficient binary serialization.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SelfEgoNode {
    /// Fixed ID for the SELF_EGO_NODE
    pub id: Uuid,
    /// Current teleological fingerprint (system state)
    pub fingerprint: Option<TeleologicalFingerprint>,
    /// System's purpose vector (alignment with north star)
    pub purpose_vector: [f32; 13],
    /// Coherence between current actions and purpose vector
    pub coherence_with_actions: f32,
    /// History of identity snapshots
    pub identity_trajectory: Vec<PurposeSnapshot>,
    /// Last update timestamp
    pub last_updated: DateTime<Utc>,
}

impl SelfEgoNode {
    /// Create a new SELF_EGO_NODE
    pub fn new() -> Self {
        // Use a fixed deterministic UUID for system identity
        let id = Uuid::nil(); // Special "zero" UUID for system identity

        Self {
            id,
            fingerprint: None,
            purpose_vector: [0.0; 13],
            coherence_with_actions: 0.0,
            identity_trajectory: Vec::new(),
            last_updated: Utc::now(),
        }
    }

    /// Initialize with a purpose vector
    pub fn with_purpose_vector(vector: [f32; 13]) -> Self {
        let mut ego = Self::new();
        ego.purpose_vector = vector;
        ego
    }

    /// Update system fingerprint (state snapshot)
    pub fn update_fingerprint(&mut self, fingerprint: TeleologicalFingerprint) -> CoreResult<()> {
        self.fingerprint = Some(fingerprint);
        self.last_updated = Utc::now();
        Ok(())
    }

    /// Record a purpose vector snapshot in the identity trajectory
    pub fn record_purpose_snapshot(&mut self, context: impl Into<String>) -> CoreResult<()> {
        let snapshot = PurposeSnapshot {
            vector: self.purpose_vector,
            timestamp: Utc::now(),
            context: context.into(),
        };
        self.identity_trajectory.push(snapshot);

        // Keep last 1000 snapshots for memory efficiency
        if self.identity_trajectory.len() > 1000 {
            self.identity_trajectory.remove(0);
        }

        Ok(())
    }

    /// Get the purpose vector at a specific point in history
    pub fn get_historical_purpose_vector(&self, index: usize) -> Option<[f32; 13]> {
        self.identity_trajectory.get(index).map(|s| s.vector)
    }

    /// Get most recent purpose snapshot
    pub fn get_latest_snapshot(&self) -> Option<&PurposeSnapshot> {
        self.identity_trajectory.last()
    }

    /// Update purpose_vector from a TeleologicalFingerprint's purpose alignments.
    ///
    /// Copies fingerprint.purpose_vector.alignments to self.purpose_vector,
    /// updates coherence_with_actions, and sets fingerprint reference.
    ///
    /// # Arguments
    /// * `fingerprint` - The source fingerprint containing purpose_vector.alignments
    ///
    /// # Returns
    /// * `CoreResult<()>` - Ok on success
    ///
    /// # Constitution Reference
    /// From constitution.yaml lines 365-392:
    /// - self_ego_node.fields includes: fingerprint, purpose_vector, coherence_with_actions
    /// - loop: "Retrieve->A(action,PV)->if<0.55 self_reflect->update fingerprint->store evolution"
    pub fn update_from_fingerprint(
        &mut self,
        fingerprint: &TeleologicalFingerprint,
    ) -> CoreResult<()> {
        // 1. Copy purpose_vector.alignments to self.purpose_vector
        self.purpose_vector = fingerprint.purpose_vector.alignments;

        // 2. Update coherence from fingerprint
        self.coherence_with_actions = fingerprint.purpose_vector.coherence;

        // 3. Store fingerprint reference (clone since we own the data)
        self.fingerprint = Some(fingerprint.clone());

        // 4. Update timestamp
        self.last_updated = Utc::now();

        // 5. Log for debugging
        tracing::debug!(
            "SelfEgoNode updated from fingerprint: purpose_vector[0]={:.4}, coherence={:.4}",
            self.purpose_vector[0],
            self.coherence_with_actions
        );

        Ok(())
    }
}

impl Default for SelfEgoNode {
    fn default() -> Self {
        Self::new()
    }
}
