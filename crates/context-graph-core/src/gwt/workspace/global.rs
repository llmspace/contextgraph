//! Global Workspace - Winner-Take-All memory selection
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

use crate::error::CoreResult;
use crate::neuromod::NeuromodulationManager;
use chrono::{DateTime, Duration, Utc};
use uuid::Uuid;

use super::candidate::WorkspaceCandidate;
use super::DA_INHIBITION_FACTOR;

/// Global workspace for consciousness broadcasting
#[derive(Debug)]
pub struct GlobalWorkspace {
    /// Currently active (conscious) memory
    pub active_memory: Option<Uuid>,
    /// Candidates in competition
    pub candidates: Vec<WorkspaceCandidate>,
    /// Coherence threshold for entry (default 0.8)
    pub coherence_threshold: f32,
    /// Broadcast duration in milliseconds
    pub broadcast_duration_ms: u64,
    /// Last broadcast time
    pub last_broadcast: Option<DateTime<Utc>>,
    /// History of previous winners (for dream replay)
    pub winner_history: Vec<(Uuid, DateTime<Utc>, f32)>, // (id, time, score)
}

impl GlobalWorkspace {
    /// Create a new global workspace
    pub fn new() -> Self {
        Self {
            active_memory: None,
            candidates: Vec::new(),
            coherence_threshold: 0.8,
            broadcast_duration_ms: 100,
            last_broadcast: None,
            winner_history: Vec::new(),
        }
    }

    /// Add a candidate memory to the workspace competition
    pub async fn add_candidate(&mut self, candidate: WorkspaceCandidate) -> CoreResult<()> {
        // Check if memory should enter workspace based on coherence
        if candidate.order_parameter >= self.coherence_threshold {
            self.candidates.push(candidate);
        }
        Ok(())
    }

    /// Select winning memory via winner-take-all
    ///
    /// # Algorithm
    /// 1. Filter candidates with r >= coherence_threshold
    /// 2. Rank by score = r * importance * alignment
    /// 3. Select top-1
    pub async fn select_winning_memory(
        &mut self,
        candidates: Vec<(Uuid, f32, f32, f32)>, // (id, r, importance, alignment)
    ) -> CoreResult<Option<Uuid>> {
        // Clear previous candidates
        self.candidates.clear();

        // Build candidates
        for (id, r, importance, alignment) in candidates {
            if let Ok(candidate) = WorkspaceCandidate::new(id, r, importance, alignment) {
                if candidate.order_parameter >= self.coherence_threshold {
                    self.candidates.push(candidate);
                }
            }
        }

        // Select winner
        if self.candidates.is_empty() {
            self.active_memory = None;
            return Ok(None);
        }

        // Sort by score (descending)
        self.candidates
            .sort_by(|a, b| b.score.partial_cmp(&a.score).unwrap());

        let winner = self.candidates[0].clone();
        let winner_id = winner.id;

        // Update workspace state
        self.active_memory = Some(winner_id);
        self.last_broadcast = Some(Utc::now());

        // Store in history (keep last 100)
        self.winner_history
            .push((winner_id, Utc::now(), winner.score));
        if self.winner_history.len() > 100 {
            self.winner_history.remove(0);
        }

        tracing::debug!(
            "Workspace selected memory: {:?} with score {:.3}",
            winner_id,
            winner.score
        );

        Ok(Some(winner_id))
    }

    /// Check if broadcast window is still active
    pub fn is_broadcasting(&self) -> bool {
        if let Some(last_time) = self.last_broadcast {
            let elapsed = Utc::now() - last_time;
            elapsed < Duration::milliseconds(self.broadcast_duration_ms as i64)
        } else {
            false
        }
    }

    /// Get the currently active memory (if broadcasting)
    pub fn get_active_memory(&self) -> Option<Uuid> {
        if self.is_broadcasting() {
            self.active_memory
        } else {
            None
        }
    }

    /// Get all candidates that passed coherence threshold
    pub fn get_coherent_candidates(&self) -> Vec<&WorkspaceCandidate> {
        self.candidates
            .iter()
            .filter(|c| c.order_parameter >= self.coherence_threshold)
            .collect()
    }

    /// Check for workspace conflict (multiple memories with r > 0.8)
    pub fn has_conflict(&self) -> bool {
        self.candidates
            .iter()
            .filter(|c| c.order_parameter > 0.8)
            .count()
            > 1
    }

    /// Get conflict details if present
    pub fn get_conflict_details(&self) -> Option<Vec<Uuid>> {
        if self.has_conflict() {
            Some(
                self.candidates
                    .iter()
                    .filter(|c| c.order_parameter > 0.8)
                    .map(|c| c.id)
                    .collect(),
            )
        } else {
            None
        }
    }

    /// Apply dopamine reduction to losing WTA candidates
    ///
    /// Per constitution gwt.global_workspace step 6:
    /// "Inhibit: losing candidates receive dopamine reduction"
    ///
    /// # Arguments
    /// * `winner_id` - The ID of the winning memory
    /// * `neuromod` - The neuromodulation manager to apply inhibition
    ///
    /// # Returns
    /// The number of candidates that were inhibited
    ///
    /// # Algorithm
    /// For each non-winner candidate:
    /// - inhibition_magnitude = (1.0 - candidate.score) * DA_INHIBITION_FACTOR
    /// - Calls neuromod.adjust(ModulatorType::Dopamine, -magnitude) to decrease dopamine
    pub fn inhibit_losers(
        &self,
        winner_id: Uuid,
        neuromod: &mut NeuromodulationManager,
    ) -> CoreResult<usize> {
        use crate::neuromod::state::ModulatorType;

        let mut inhibited_count = 0;

        for candidate in &self.candidates {
            if candidate.id != winner_id {
                // Calculate inhibition magnitude based on how far from winning
                // Lower score = stronger inhibition
                let inhibition_magnitude = (1.0 - candidate.score) * DA_INHIBITION_FACTOR;

                // Apply dopamine reduction (negative delta)
                neuromod.adjust(ModulatorType::Dopamine, -inhibition_magnitude)?;

                tracing::debug!(
                    "Inhibited loser memory {:?}: score={:.3}, inhibition={:.4}",
                    candidate.id,
                    candidate.score,
                    inhibition_magnitude
                );

                inhibited_count += 1;
            }
        }

        tracing::debug!(
            "WTA inhibition complete: {} losers inhibited, winner={:?}",
            inhibited_count,
            winner_id
        );

        Ok(inhibited_count)
    }
}

impl Default for GlobalWorkspace {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
#[path = "global_tests.rs"]
mod tests;
