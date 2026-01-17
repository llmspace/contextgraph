//! Prediction history for autonomous learner.
//!
//! TASK-004: Implements prediction tracking with TTL eviction.
//! Per EC-AUTO-03: 24-hour TTL for prediction history.
//!
//! This enables `observe_outcome` to look up original predicted values
//! instead of using hardcoded defaults.

use std::collections::HashMap;
use std::time::{Duration, Instant};

use tracing::{debug, warn};
use uuid::Uuid;

/// TTL for predictions in the history (24 hours per EC-AUTO-03).
const PREDICTION_TTL: Duration = Duration::from_secs(24 * 60 * 60);

/// Entry in the prediction history.
#[derive(Debug, Clone)]
pub struct PredictionEntry {
    /// The predicted value (0.0-1.0)
    pub predicted_value: f32,
    /// When the prediction was made
    pub created_at: Instant,
    /// Domain of the prediction (optional)
    pub domain: Option<String>,
    /// Additional context (optional)
    pub context: Option<String>,
}

impl PredictionEntry {
    /// Check if this entry has expired.
    pub fn is_expired(&self) -> bool {
        self.created_at.elapsed() > PREDICTION_TTL
    }

    /// Get the age of this prediction in seconds.
    pub fn age_secs(&self) -> u64 {
        self.created_at.elapsed().as_secs()
    }
}

/// Prediction history with TTL eviction.
///
/// TASK-004: Stores predictions by ID for later lookup by `observe_outcome`.
/// Entries are automatically evicted after 24 hours (EC-AUTO-03).
///
/// Thread-safety: This struct is NOT internally synchronized.
/// Wrap in `Arc<RwLock<PredictionHistory>>` for concurrent access.
#[derive(Debug)]
pub struct PredictionHistory {
    /// Predictions indexed by ID
    entries: HashMap<Uuid, PredictionEntry>,
    /// Last cleanup timestamp
    last_cleanup: Instant,
    /// Cleanup interval (run cleanup at most once per this duration)
    cleanup_interval: Duration,
}

impl Default for PredictionHistory {
    fn default() -> Self {
        Self::new()
    }
}

impl PredictionHistory {
    /// Create a new empty prediction history.
    pub fn new() -> Self {
        Self {
            entries: HashMap::new(),
            last_cleanup: Instant::now(),
            // Run cleanup at most once per hour
            cleanup_interval: Duration::from_secs(60 * 60),
        }
    }

    /// Store a prediction in the history.
    ///
    /// # Arguments
    /// * `prediction_id` - Unique ID for this prediction
    /// * `predicted_value` - The predicted value (0.0-1.0)
    /// * `domain` - Optional domain (e.g., "Code", "Medical")
    /// * `context` - Optional context string
    ///
    /// # Returns
    /// The prediction ID (for confirmation)
    pub fn store(
        &mut self,
        prediction_id: Uuid,
        predicted_value: f32,
        domain: Option<String>,
        context: Option<String>,
    ) -> Uuid {
        // Maybe cleanup expired entries
        self.maybe_cleanup();

        let entry = PredictionEntry {
            predicted_value: predicted_value.clamp(0.0, 1.0),
            created_at: Instant::now(),
            domain,
            context,
        };

        debug!(
            prediction_id = %prediction_id,
            predicted_value = predicted_value,
            "PredictionHistory: Stored prediction"
        );

        self.entries.insert(prediction_id, entry);
        prediction_id
    }

    /// Look up a prediction by ID.
    ///
    /// # Arguments
    /// * `prediction_id` - The ID to look up
    ///
    /// # Returns
    /// The prediction entry if found and not expired, None otherwise.
    pub fn get(&self, prediction_id: &Uuid) -> Option<&PredictionEntry> {
        self.entries.get(prediction_id).filter(|e| !e.is_expired())
    }

    /// Look up and remove a prediction by ID.
    ///
    /// # Arguments
    /// * `prediction_id` - The ID to look up and remove
    ///
    /// # Returns
    /// The prediction entry if found and not expired, None otherwise.
    /// The entry is removed from history after lookup.
    pub fn take(&mut self, prediction_id: &Uuid) -> Option<PredictionEntry> {
        // Maybe cleanup expired entries
        self.maybe_cleanup();

        match self.entries.remove(prediction_id) {
            Some(entry) if !entry.is_expired() => {
                debug!(
                    prediction_id = %prediction_id,
                    age_secs = entry.age_secs(),
                    "PredictionHistory: Retrieved prediction"
                );
                Some(entry)
            }
            Some(entry) => {
                warn!(
                    prediction_id = %prediction_id,
                    age_secs = entry.age_secs(),
                    "PredictionHistory: Prediction expired"
                );
                None
            }
            None => {
                debug!(
                    prediction_id = %prediction_id,
                    "PredictionHistory: Prediction not found"
                );
                None
            }
        }
    }

    /// Get the number of entries in the history (including expired).
    pub fn len(&self) -> usize {
        self.entries.len()
    }

    /// Check if the history is empty.
    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }

    /// Remove all expired entries.
    ///
    /// # Returns
    /// Number of entries removed.
    pub fn cleanup(&mut self) -> usize {
        let before = self.entries.len();
        self.entries.retain(|_, entry| !entry.is_expired());
        let removed = before - self.entries.len();

        if removed > 0 {
            debug!(
                removed = removed,
                remaining = self.entries.len(),
                "PredictionHistory: Cleaned up expired entries"
            );
        }

        self.last_cleanup = Instant::now();
        removed
    }

    /// Run cleanup if enough time has passed since last cleanup.
    fn maybe_cleanup(&mut self) {
        if self.last_cleanup.elapsed() > self.cleanup_interval {
            self.cleanup();
        }
    }

    /// Get the TTL duration for predictions.
    pub fn ttl() -> Duration {
        PREDICTION_TTL
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_store_and_get() {
        let mut history = PredictionHistory::new();
        let id = Uuid::new_v4();

        history.store(id, 0.75, Some("Code".to_string()), None);

        let entry = history.get(&id);
        assert!(entry.is_some());
        assert!((entry.unwrap().predicted_value - 0.75).abs() < f32::EPSILON);
    }

    #[test]
    fn test_take_removes_entry() {
        let mut history = PredictionHistory::new();
        let id = Uuid::new_v4();

        history.store(id, 0.8, None, None);
        assert_eq!(history.len(), 1);

        let entry = history.take(&id);
        assert!(entry.is_some());
        assert_eq!(history.len(), 0);

        // Second take should return None
        let entry2 = history.take(&id);
        assert!(entry2.is_none());
    }

    #[test]
    fn test_clamping() {
        let mut history = PredictionHistory::new();
        let id1 = Uuid::new_v4();
        let id2 = Uuid::new_v4();

        history.store(id1, 1.5, None, None); // Should clamp to 1.0
        history.store(id2, -0.5, None, None); // Should clamp to 0.0

        assert!((history.get(&id1).unwrap().predicted_value - 1.0).abs() < f32::EPSILON);
        assert!((history.get(&id2).unwrap().predicted_value - 0.0).abs() < f32::EPSILON);
    }

    #[test]
    fn test_missing_entry() {
        let history = PredictionHistory::new();
        let id = Uuid::new_v4();

        assert!(history.get(&id).is_none());
    }
}
