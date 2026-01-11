//! PurposeVectorHistory - Ring buffer for purpose vector snapshots
//!
//! Manages purpose vector history for identity continuity calculation.

use chrono::Utc;
use serde::{Deserialize, Serialize};
use std::collections::VecDeque;

use super::types::{PurposeSnapshot, MAX_PV_HISTORY_SIZE};

/// Trait for purpose vector history operations
///
/// Enables abstraction for testing and alternative implementations.
pub trait PurposeVectorHistoryProvider {
    /// Push a new purpose vector with context
    ///
    /// # Arguments
    /// * `pv` - The 13D purpose vector (must be exactly 13 elements)
    /// * `context` - Description of what triggered this snapshot
    ///
    /// # Returns
    /// The previous purpose vector, if any existed (for IC calculation)
    fn push(&mut self, pv: [f32; 13], context: impl Into<String>) -> Option<[f32; 13]>;

    /// Get the current (most recent) purpose vector
    ///
    /// # Returns
    /// - `Some(&[f32; 13])` if history has at least one entry
    /// - `None` if history is empty
    fn current(&self) -> Option<&[f32; 13]>;

    /// Get the previous purpose vector (for IC calculation)
    ///
    /// # Returns
    /// - `Some(&[f32; 13])` if history has at least two entries
    /// - `None` if history has 0 or 1 entries
    fn previous(&self) -> Option<&[f32; 13]>;

    /// Get both current and previous for IC calculation
    ///
    /// # Returns
    /// - `Some((current, Some(previous)))` if len >= 2
    /// - `Some((current, None))` if len == 1 (first vector)
    /// - `None` if empty
    fn current_and_previous(&self) -> Option<(&[f32; 13], Option<&[f32; 13]>)>;

    /// Get the number of snapshots in history
    fn len(&self) -> usize;

    /// Check if history is empty
    fn is_empty(&self) -> bool;

    /// Check if this is the first vector (exactly one entry, no previous)
    ///
    /// # Edge Case
    /// Per EC-IDENTITY-01: First vector defaults to IC = 1.0 (Healthy)
    fn is_first_vector(&self) -> bool;
}

/// Manages purpose vector history for identity continuity calculation
///
/// Provides O(1) access to current and previous purpose vectors,
/// handling the edge case of first vector (no previous).
///
/// # Constitution Reference
/// - self_ego_node.identity_trajectory: max 1000 snapshots
/// - IC = cos(PV_t, PV_{t-1}) x r(t) requires consecutive PV access
///
/// # Memory Management
/// Uses FIFO eviction when reaching MAX_PV_HISTORY_SIZE.
/// VecDeque ensures O(1) push_back and pop_front.
///
/// # Error Handling
/// This type does NOT panic. All operations return Option or Result.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PurposeVectorHistory {
    /// Ring buffer of purpose snapshots (VecDeque for O(1) operations)
    history: VecDeque<PurposeSnapshot>,
    /// Maximum history size (default: 1000)
    pub max_size: usize,
}

impl PurposeVectorHistory {
    /// Create new history with default max size (1000)
    pub fn new() -> Self {
        Self::with_max_size(MAX_PV_HISTORY_SIZE)
    }

    /// Create with custom max size
    ///
    /// # Arguments
    /// * `max_size` - Maximum entries before FIFO eviction
    ///
    /// # Notes
    /// max_size of 0 means no eviction limit.
    pub fn with_max_size(max_size: usize) -> Self {
        Self {
            // Pre-allocate up to 1024 to avoid reallocs
            history: VecDeque::with_capacity(max_size.min(1024)),
            max_size,
        }
    }

    /// Get read access to full history (for diagnostics)
    pub fn history(&self) -> &VecDeque<PurposeSnapshot> {
        &self.history
    }
}

impl Default for PurposeVectorHistory {
    fn default() -> Self {
        Self::new()
    }
}

impl PurposeVectorHistoryProvider for PurposeVectorHistory {
    fn push(&mut self, pv: [f32; 13], context: impl Into<String>) -> Option<[f32; 13]> {
        // Capture previous BEFORE pushing
        let previous = self.current().copied();

        // FIFO eviction if at capacity
        if self.max_size > 0 && self.history.len() >= self.max_size {
            self.history.pop_front(); // O(1) with VecDeque
        }

        // Add new snapshot
        self.history.push_back(PurposeSnapshot {
            vector: pv,
            timestamp: Utc::now(),
            context: context.into(),
        });

        previous
    }

    fn current(&self) -> Option<&[f32; 13]> {
        self.history.back().map(|s| &s.vector)
    }

    fn previous(&self) -> Option<&[f32; 13]> {
        if self.history.len() < 2 {
            return None;
        }
        // VecDeque indexing is O(1)
        self.history.get(self.history.len() - 2).map(|s| &s.vector)
    }

    fn current_and_previous(&self) -> Option<(&[f32; 13], Option<&[f32; 13]>)> {
        self.current().map(|curr| (curr, self.previous()))
    }

    fn len(&self) -> usize {
        self.history.len()
    }

    fn is_empty(&self) -> bool {
        self.history.is_empty()
    }

    fn is_first_vector(&self) -> bool {
        self.history.len() == 1
    }
}
