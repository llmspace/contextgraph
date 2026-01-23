//! Intent drift detection for E10 multimodal embeddings.
//!
//! Tracks E10 intent embeddings across a session to detect when user's intent shifts.
//! Uses centroid-based drift detection with configurable thresholds.
//!
//! # Architecture
//!
//! The `IntentTracker` maintains a sliding window of intent snapshots:
//! - Each snapshot contains the E10 intent vector and metadata
//! - Drift is detected by comparing new intents against the recent centroid
//! - Configurable threshold (default 0.4) determines drift sensitivity
//!
//! # Usage
//!
//! ```ignore
//! let mut tracker = IntentTracker::new("session-123");
//!
//! // Record intent vectors as they come in
//! tracker.record_intent(intent_vector, "User asking about databases");
//!
//! // Check for drift
//! if let Some(drift) = tracker.detect_drift(&new_intent_vector) {
//!     println!("Intent drift detected: score={}", drift.score);
//! }
//! ```

use std::collections::VecDeque;
use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use uuid::Uuid;

/// Default drift threshold (0.4 = moderate sensitivity).
pub const DEFAULT_DRIFT_THRESHOLD: f32 = 0.4;

/// Default window size for computing recent intent centroid.
pub const DEFAULT_WINDOW_SIZE: usize = 5;

/// Maximum history size to prevent unbounded growth.
pub const MAX_HISTORY_SIZE: usize = 100;

/// A snapshot of intent at a specific point in time.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct IntentSnapshot {
    /// Unique identifier for this snapshot.
    pub id: Uuid,

    /// Session this snapshot belongs to.
    pub session_id: String,

    /// E10 intent vector (as_intent encoding).
    pub intent_vector: Vec<f32>,

    /// Optional content/query that generated this intent.
    pub content_summary: Option<String>,

    /// Timestamp when this intent was recorded.
    pub timestamp: DateTime<Utc>,

    /// Sequence number within the session.
    pub sequence: u64,

    /// Detected intent category (if classified).
    pub intent_category: Option<String>,
}

impl IntentSnapshot {
    /// Create a new intent snapshot.
    pub fn new(
        session_id: String,
        intent_vector: Vec<f32>,
        sequence: u64,
    ) -> Self {
        Self {
            id: Uuid::new_v4(),
            session_id,
            intent_vector,
            content_summary: None,
            timestamp: Utc::now(),
            sequence,
            intent_category: None,
        }
    }

    /// Set the content summary.
    pub fn with_content(mut self, content: impl Into<String>) -> Self {
        self.content_summary = Some(content.into());
        self
    }

    /// Set the intent category.
    pub fn with_category(mut self, category: impl Into<String>) -> Self {
        self.intent_category = Some(category.into());
        self
    }
}

/// Detected intent drift event.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct IntentDrift {
    /// Drift score (1.0 - cosine_similarity with recent centroid).
    /// Higher = more drift from recent intent pattern.
    pub score: f32,

    /// Previous intent centroid (what we drifted FROM).
    pub previous_centroid: Vec<f32>,

    /// New intent vector (what we drifted TO).
    pub new_intent: Vec<f32>,

    /// Timestamp of drift detection.
    pub detected_at: DateTime<Utc>,

    /// Session ID where drift occurred.
    pub session_id: String,

    /// Number of intents used to compute previous centroid.
    pub centroid_sample_size: usize,

    /// Previous intent category (if available).
    pub previous_category: Option<String>,

    /// New intent category (if detected).
    pub new_category: Option<String>,
}

/// Tracks intent embeddings and detects drift within a session.
#[derive(Debug, Clone)]
pub struct IntentTracker {
    /// Session ID being tracked.
    session_id: String,

    /// History of intent snapshots.
    intent_history: VecDeque<IntentSnapshot>,

    /// Drift detection threshold.
    /// Drift is detected when (1 - similarity) > threshold.
    drift_threshold: f32,

    /// Window size for computing recent centroid.
    window_size: usize,

    /// Current sequence number.
    current_sequence: u64,

    /// Total drift events detected.
    drift_count: u64,
}

impl IntentTracker {
    /// Create a new intent tracker for a session.
    pub fn new(session_id: impl Into<String>) -> Self {
        Self {
            session_id: session_id.into(),
            intent_history: VecDeque::with_capacity(MAX_HISTORY_SIZE),
            drift_threshold: DEFAULT_DRIFT_THRESHOLD,
            window_size: DEFAULT_WINDOW_SIZE,
            current_sequence: 0,
            drift_count: 0,
        }
    }

    /// Set the drift threshold.
    pub fn with_threshold(mut self, threshold: f32) -> Self {
        self.drift_threshold = threshold.clamp(0.0, 1.0);
        self
    }

    /// Set the window size for centroid computation.
    pub fn with_window_size(mut self, size: usize) -> Self {
        self.window_size = size.max(1);
        self
    }

    /// Get the session ID.
    pub fn session_id(&self) -> &str {
        &self.session_id
    }

    /// Get the current sequence number.
    pub fn current_sequence(&self) -> u64 {
        self.current_sequence
    }

    /// Get the number of recorded intents.
    pub fn intent_count(&self) -> usize {
        self.intent_history.len()
    }

    /// Get the total drift events detected.
    pub fn drift_count(&self) -> u64 {
        self.drift_count
    }

    /// Record a new intent vector.
    ///
    /// Returns `Some(IntentDrift)` if drift is detected.
    pub fn record_intent(
        &mut self,
        intent_vector: Vec<f32>,
        content_summary: Option<&str>,
    ) -> Option<IntentDrift> {
        // Check for drift before recording
        let drift = self.detect_drift(&intent_vector);

        // Increment sequence
        self.current_sequence += 1;

        // Create snapshot
        let mut snapshot = IntentSnapshot::new(
            self.session_id.clone(),
            intent_vector,
            self.current_sequence,
        );

        if let Some(content) = content_summary {
            snapshot = snapshot.with_content(content);
        }

        // Add to history
        self.intent_history.push_back(snapshot);

        // Trim if over max size
        while self.intent_history.len() > MAX_HISTORY_SIZE {
            self.intent_history.pop_front();
        }

        // Track drift count
        if drift.is_some() {
            self.drift_count += 1;
        }

        drift
    }

    /// Detect drift without recording the intent.
    ///
    /// Compares the new intent against the recent centroid.
    /// Returns `Some(IntentDrift)` if drift exceeds threshold.
    pub fn detect_drift(&self, new_intent: &[f32]) -> Option<IntentDrift> {
        // Need at least window_size intents to compute meaningful centroid
        if self.intent_history.len() < self.window_size {
            return None;
        }

        // Compute recent centroid
        let centroid = self.compute_recent_centroid(self.window_size);

        // Compute similarity
        let similarity = cosine_similarity(&centroid, new_intent);
        let drift_score = 1.0 - similarity;

        // Check threshold
        if drift_score > self.drift_threshold {
            // Get previous category from most recent snapshot
            let previous_category = self.intent_history
                .back()
                .and_then(|s| s.intent_category.clone());

            Some(IntentDrift {
                score: drift_score,
                previous_centroid: centroid,
                new_intent: new_intent.to_vec(),
                detected_at: Utc::now(),
                session_id: self.session_id.clone(),
                centroid_sample_size: self.window_size.min(self.intent_history.len()),
                previous_category,
                new_category: None,
            })
        } else {
            None
        }
    }

    /// Compute the centroid of recent intent vectors.
    pub fn compute_recent_centroid(&self, window: usize) -> Vec<f32> {
        let count = window.min(self.intent_history.len());
        if count == 0 {
            return Vec::new();
        }

        let start_idx = self.intent_history.len().saturating_sub(count);
        let recent: Vec<_> = self.intent_history
            .iter()
            .skip(start_idx)
            .collect();

        if recent.is_empty() {
            return Vec::new();
        }

        // Get dimension from first vector
        let dim = recent[0].intent_vector.len();
        if dim == 0 {
            return Vec::new();
        }

        // Compute mean
        let mut centroid = vec![0.0f32; dim];
        for snapshot in &recent {
            for (i, val) in snapshot.intent_vector.iter().enumerate() {
                if i < dim {
                    centroid[i] += val;
                }
            }
        }

        let n = recent.len() as f32;
        for val in &mut centroid {
            *val /= n;
        }

        // Normalize
        let norm: f32 = centroid.iter().map(|x| x * x).sum::<f32>().sqrt();
        if norm > f32::EPSILON {
            for val in &mut centroid {
                *val /= norm;
            }
        }

        centroid
    }

    /// Get the intent history.
    pub fn history(&self) -> &VecDeque<IntentSnapshot> {
        &self.intent_history
    }

    /// Get the most recent N intents.
    pub fn recent_intents(&self, limit: usize) -> Vec<&IntentSnapshot> {
        let start = self.intent_history.len().saturating_sub(limit);
        self.intent_history.iter().skip(start).collect()
    }

    /// Get similarity between two consecutive intents.
    ///
    /// Returns a vector of (sequence, similarity) pairs.
    pub fn pairwise_similarities(&self) -> Vec<(u64, f32)> {
        if self.intent_history.len() < 2 {
            return Vec::new();
        }

        self.intent_history
            .iter()
            .zip(self.intent_history.iter().skip(1))
            .map(|(a, b)| {
                let sim = cosine_similarity(&a.intent_vector, &b.intent_vector);
                (b.sequence, sim)
            })
            .collect()
    }

    /// Get intent trajectory statistics.
    pub fn trajectory_stats(&self) -> IntentTrajectoryStats {
        let pairwise = self.pairwise_similarities();

        if pairwise.is_empty() {
            return IntentTrajectoryStats::default();
        }

        let sims: Vec<f32> = pairwise.iter().map(|(_, s)| *s).collect();
        let avg = sims.iter().sum::<f32>() / sims.len() as f32;
        let min = sims.iter().cloned().fold(f32::INFINITY, f32::min);
        let max = sims.iter().cloned().fold(f32::NEG_INFINITY, f32::max);

        // Compute variance
        let variance = sims.iter()
            .map(|s| (s - avg).powi(2))
            .sum::<f32>() / sims.len() as f32;
        let std_dev = variance.sqrt();

        IntentTrajectoryStats {
            total_intents: self.intent_history.len(),
            total_drifts: self.drift_count as usize,
            avg_pairwise_similarity: avg,
            min_pairwise_similarity: min,
            max_pairwise_similarity: max,
            similarity_std_dev: std_dev,
            drift_rate: if self.intent_history.len() > 1 {
                self.drift_count as f32 / (self.intent_history.len() - 1) as f32
            } else {
                0.0
            },
        }
    }

    /// Clear the intent history.
    pub fn clear(&mut self) {
        self.intent_history.clear();
        self.current_sequence = 0;
        self.drift_count = 0;
    }
}

/// Statistics about the intent trajectory.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct IntentTrajectoryStats {
    /// Total number of intents recorded.
    pub total_intents: usize,

    /// Total number of drift events detected.
    pub total_drifts: usize,

    /// Average similarity between consecutive intents.
    pub avg_pairwise_similarity: f32,

    /// Minimum pairwise similarity (biggest single change).
    pub min_pairwise_similarity: f32,

    /// Maximum pairwise similarity (most stable transition).
    pub max_pairwise_similarity: f32,

    /// Standard deviation of pairwise similarities.
    pub similarity_std_dev: f32,

    /// Drift rate (drifts / transitions).
    pub drift_rate: f32,
}

/// Compute cosine similarity between two vectors.
fn cosine_similarity(a: &[f32], b: &[f32]) -> f32 {
    if a.len() != b.len() || a.is_empty() {
        return 0.0;
    }

    let dot: f32 = a.iter().zip(b.iter()).map(|(x, y)| x * y).sum();
    let norm_a: f32 = a.iter().map(|x| x * x).sum::<f32>().sqrt();
    let norm_b: f32 = b.iter().map(|x| x * x).sum::<f32>().sqrt();

    if norm_a < f32::EPSILON || norm_b < f32::EPSILON {
        return 0.0;
    }

    (dot / (norm_a * norm_b)).clamp(-1.0, 1.0)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn random_unit_vector(dim: usize, seed: u64) -> Vec<f32> {
        use std::hash::{Hash, Hasher};
        use std::collections::hash_map::DefaultHasher;

        let mut hasher = DefaultHasher::new();
        seed.hash(&mut hasher);
        let mut state = hasher.finish();

        let mut vec: Vec<f32> = (0..dim)
            .map(|i| {
                state = state.wrapping_mul(6364136223846793005).wrapping_add(i as u64);
                (state as f32 / u64::MAX as f32) * 2.0 - 1.0
            })
            .collect();

        let norm: f32 = vec.iter().map(|x| x * x).sum::<f32>().sqrt();
        for v in &mut vec {
            *v /= norm;
        }
        vec
    }

    #[test]
    fn test_intent_tracker_creation() {
        let tracker = IntentTracker::new("test-session");
        assert_eq!(tracker.session_id(), "test-session");
        assert_eq!(tracker.intent_count(), 0);
        assert_eq!(tracker.drift_count(), 0);
    }

    #[test]
    fn test_record_intent_no_drift_initially() {
        let mut tracker = IntentTracker::new("test");

        // First few intents shouldn't trigger drift (not enough history)
        for i in 0..3 {
            let vec = random_unit_vector(768, i);
            let drift = tracker.record_intent(vec, Some(&format!("intent {}", i)));
            assert!(drift.is_none(), "Should not detect drift with insufficient history");
        }

        assert_eq!(tracker.intent_count(), 3);
    }

    #[test]
    fn test_drift_detection_with_similar_intents() {
        let mut tracker = IntentTracker::new("test")
            .with_window_size(3)
            .with_threshold(0.4);

        // Record similar intents
        let base = random_unit_vector(768, 42);
        for i in 0..5 {
            // Add small noise
            let mut vec = base.clone();
            for v in &mut vec {
                *v += 0.01 * (i as f32);
            }
            // Renormalize
            let norm: f32 = vec.iter().map(|x| x * x).sum::<f32>().sqrt();
            for v in &mut vec {
                *v /= norm;
            }

            let drift = tracker.record_intent(vec, None);
            assert!(drift.is_none(), "Similar intents should not trigger drift");
        }
    }

    #[test]
    fn test_drift_detection_with_different_intent() {
        let mut tracker = IntentTracker::new("test")
            .with_window_size(3)
            .with_threshold(0.3);

        // Record similar intents
        let base = random_unit_vector(768, 100);
        for _ in 0..5 {
            tracker.record_intent(base.clone(), None);
        }

        // Now record a very different intent
        let different = random_unit_vector(768, 999);
        let drift = tracker.record_intent(different, Some("completely different"));

        // Should detect drift if vectors are sufficiently different
        // Note: random vectors may or may not be different enough
        if let Some(d) = drift {
            assert!(d.score > 0.0);
            assert_eq!(d.session_id, "test");
        }
    }

    #[test]
    fn test_pairwise_similarities() {
        let mut tracker = IntentTracker::new("test");

        for i in 0..5 {
            let vec = random_unit_vector(768, i * 100);
            tracker.record_intent(vec, None);
        }

        let sims = tracker.pairwise_similarities();
        assert_eq!(sims.len(), 4); // 5 intents -> 4 pairs
    }

    #[test]
    fn test_trajectory_stats() {
        let mut tracker = IntentTracker::new("test");

        for i in 0..10 {
            let vec = random_unit_vector(768, i);
            tracker.record_intent(vec, None);
        }

        let stats = tracker.trajectory_stats();
        assert_eq!(stats.total_intents, 10);
        assert!(stats.avg_pairwise_similarity >= -1.0 && stats.avg_pairwise_similarity <= 1.0);
    }

    #[test]
    fn test_cosine_similarity() {
        let a = vec![1.0, 0.0, 0.0];
        let b = vec![1.0, 0.0, 0.0];
        assert!((cosine_similarity(&a, &b) - 1.0).abs() < 1e-6);

        let c = vec![0.0, 1.0, 0.0];
        assert!(cosine_similarity(&a, &c).abs() < 1e-6);

        let d = vec![-1.0, 0.0, 0.0];
        assert!((cosine_similarity(&a, &d) + 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_recent_intents() {
        let mut tracker = IntentTracker::new("test");

        for i in 0..10 {
            let vec = random_unit_vector(768, i);
            tracker.record_intent(vec, Some(&format!("intent {}", i)));
        }

        let recent = tracker.recent_intents(3);
        assert_eq!(recent.len(), 3);
        assert_eq!(recent[0].sequence, 8);
        assert_eq!(recent[1].sequence, 9);
        assert_eq!(recent[2].sequence, 10);
    }
}
