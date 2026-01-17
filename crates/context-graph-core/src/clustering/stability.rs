//! Portfolio-level topic stability tracking for dream triggers.
//!
//! DISTINCT FROM TopicStability (per-topic metrics in topic.rs).
//! This module tracks the entire topic portfolio over time.
//!
//! # Dream Trigger Conditions (per constitution AP-70)
//!
//! Dream consolidation triggers when EITHER:
//! 1. entropy > 0.7 AND churn > 0.5 (both simultaneously)
//! 2. entropy > 0.7 for 5+ continuous minutes
//!
//! # Churn Calculation
//!
//! churn = |symmetric_difference| / |union|
//! where symmetric_difference = topics_added + topics_removed

use std::collections::{HashSet, VecDeque};

use chrono::{DateTime, Duration, Utc};
use uuid::Uuid;

use super::topic::Topic;

/// Default churn threshold for dream trigger (0.5 per constitution).
pub const DEFAULT_CHURN_THRESHOLD: f32 = 0.5;

/// Default entropy threshold for dream trigger (0.7 per constitution).
pub const DEFAULT_ENTROPY_THRESHOLD: f32 = 0.7;

/// High entropy must persist for 5 minutes to trigger dream (300 seconds).
pub const DEFAULT_ENTROPY_DURATION_SECS: u64 = 300;

/// Snapshots retained for 24 hours.
pub const SNAPSHOT_RETENTION_HOURS: i64 = 24;

/// Snapshot of topic portfolio at a point in time.
///
/// Used for computing churn between snapshots. Only stores topic IDs
/// (not full TopicProfile) since churn = topics added/removed.
#[derive(Debug, Clone)]
pub struct TopicSnapshot {
    /// When this snapshot was taken.
    pub timestamp: DateTime<Utc>,
    /// Topic IDs present at this time.
    pub topic_ids: Vec<Uuid>,
    /// Total member count across all topics.
    pub total_members: usize,
}

impl TopicSnapshot {
    /// Create snapshot from current topic portfolio.
    pub fn from_topics(topics: &[Topic]) -> Self {
        Self::at_time(topics, Utc::now())
    }

    /// Create snapshot with specific timestamp (for testing).
    #[cfg(test)]
    pub fn with_timestamp(topics: &[Topic], timestamp: DateTime<Utc>) -> Self {
        Self::at_time(topics, timestamp)
    }

    /// Internal helper to create a snapshot at a given time.
    fn at_time(topics: &[Topic], timestamp: DateTime<Utc>) -> Self {
        Self {
            timestamp,
            topic_ids: topics.iter().map(|t| t.id).collect(),
            total_members: topics.iter().map(|t| t.member_count()).sum(),
        }
    }
}

/// Tracks portfolio-level topic stability and dream triggers.
///
/// This is DISTINCT from TopicStability (per-topic metrics).
/// TopicStabilityTracker monitors the entire topic portfolio:
/// - Takes periodic snapshots of all topics
/// - Computes churn rate (topics appearing/disappearing)
/// - Tracks high-entropy duration for dream triggers
///
/// # Dream Trigger Conditions
///
/// Per constitution AP-70, dream triggers when:
/// 1. entropy > 0.7 AND churn > 0.5 (simultaneous)
/// 2. entropy > 0.7 for 5+ continuous minutes
#[derive(Debug)]
pub struct TopicStabilityTracker {
    /// Historical snapshots (last 24 hours).
    snapshots: VecDeque<TopicSnapshot>,
    /// Most recent computed churn rate.
    current_churn: f32,
    /// When high entropy started (for duration tracking).
    high_entropy_start: Option<DateTime<Utc>>,
    /// Churn threshold for dream trigger (default 0.5).
    churn_threshold: f32,
    /// Entropy threshold for dream trigger (default 0.7).
    entropy_threshold: f32,
    /// Required high-entropy duration in seconds (default 300).
    entropy_duration_secs: u64,
    /// History of churn calculations with timestamps.
    churn_history: VecDeque<(DateTime<Utc>, f32)>,
}

impl Default for TopicStabilityTracker {
    fn default() -> Self {
        Self::new()
    }
}

impl TopicStabilityTracker {
    /// Create with default configuration per constitution.
    ///
    /// - churn_threshold: 0.5
    /// - entropy_threshold: 0.7
    /// - entropy_duration: 5 minutes (300 seconds)
    pub fn new() -> Self {
        Self {
            snapshots: VecDeque::new(),
            current_churn: 0.0,
            high_entropy_start: None,
            churn_threshold: DEFAULT_CHURN_THRESHOLD,
            entropy_threshold: DEFAULT_ENTROPY_THRESHOLD,
            entropy_duration_secs: DEFAULT_ENTROPY_DURATION_SECS,
            churn_history: VecDeque::new(),
        }
    }

    /// Create with custom thresholds.
    ///
    /// # Arguments
    /// * `churn` - Churn threshold (clamped to 0.0..=1.0)
    /// * `entropy` - Entropy threshold (clamped to 0.0..=1.0)
    /// * `duration_secs` - High-entropy duration required
    pub fn with_thresholds(churn: f32, entropy: f32, duration_secs: u64) -> Self {
        Self {
            snapshots: VecDeque::new(),
            current_churn: 0.0,
            high_entropy_start: None,
            churn_threshold: churn.clamp(0.0, 1.0),
            entropy_threshold: entropy.clamp(0.0, 1.0),
            entropy_duration_secs: duration_secs,
            churn_history: VecDeque::new(),
        }
    }

    /// Take a snapshot of current topic portfolio.
    ///
    /// Stores topic IDs and member counts. Old snapshots (>24h) are cleaned.
    pub fn take_snapshot(&mut self, topics: &[Topic]) {
        let snapshot = TopicSnapshot::from_topics(topics);
        self.snapshots.push_back(snapshot);
        self.cleanup_old_snapshots();
    }

    /// Compute churn by comparing current state to ~1 hour ago.
    ///
    /// churn = |symmetric_difference| / |union|
    /// where symmetric_difference = topics_added + topics_removed
    ///
    /// # Returns
    /// Churn rate in range [0.0, 1.0] where:
    /// - 0.0 = no change (stable)
    /// - 1.0 = complete turnover
    pub fn track_churn(&mut self) -> f32 {
        let now = Utc::now();
        let one_hour_ago = now - Duration::hours(1);

        // Find closest snapshot to 1 hour ago (iterate from back for efficiency)
        let old_snapshot = self
            .snapshots
            .iter()
            .rev()
            .find(|s| s.timestamp <= one_hour_ago);

        // Compute churn if we have both old and current snapshots
        let churn = match (old_snapshot, self.snapshots.back()) {
            (Some(old), Some(current)) => self.compute_churn(old, current),
            _ => 0.0,
        };

        self.current_churn = churn;
        self.churn_history.push_back((now, churn));
        self.cleanup_old_churn_history(now);

        churn
    }

    /// Compute churn between two snapshots using Jaccard distance formula.
    ///
    /// churn = |symmetric_difference| / |union|
    ///
    /// This is equivalent to 1 - Jaccard similarity.
    fn compute_churn(&self, old: &TopicSnapshot, current: &TopicSnapshot) -> f32 {
        let old_ids: HashSet<_> = old.topic_ids.iter().collect();
        let current_ids: HashSet<_> = current.topic_ids.iter().collect();

        let union_count = old_ids.union(&current_ids).count();
        if union_count == 0 {
            return 0.0;
        }

        let symmetric_diff = old_ids.symmetric_difference(&current_ids).count();
        let churn = symmetric_diff as f32 / union_count as f32;

        // Guard against NaN/Infinity per AP-10
        if churn.is_nan() || churn.is_infinite() {
            0.0
        } else {
            churn.clamp(0.0, 1.0)
        }
    }

    /// Check if dream consolidation should be triggered.
    ///
    /// Per constitution AP-70, triggers when EITHER:
    /// 1. entropy > 0.7 AND churn > 0.5 (both simultaneously)
    /// 2. entropy > 0.7 for 5+ continuous minutes
    ///
    /// # Arguments
    /// * `entropy` - Current system entropy [0.0, 1.0]
    ///
    /// # Returns
    /// true if dream should be triggered
    pub fn check_dream_trigger(&mut self, entropy: f32) -> bool {
        let now = Utc::now();
        let is_high_entropy = entropy > self.entropy_threshold;

        // Track high-entropy duration: start timer or reset based on current entropy
        self.high_entropy_start = match (is_high_entropy, self.high_entropy_start) {
            (true, None) => Some(now),
            (true, Some(start)) => Some(start),
            (false, _) => None,
        };

        // Condition 1: High entropy AND high churn (per AP-70)
        if is_high_entropy && self.current_churn > self.churn_threshold {
            return true;
        }

        // Condition 2: Sustained high entropy for required duration
        if let Some(start) = self.high_entropy_start {
            let duration_secs = now.signed_duration_since(start).num_seconds();
            if duration_secs >= 0 && duration_secs as u64 >= self.entropy_duration_secs {
                return true;
            }
        }

        false
    }

    /// Get current churn rate.
    #[inline]
    pub fn current_churn(&self) -> f32 {
        self.current_churn
    }

    /// Set current churn rate (for testing/simulation).
    ///
    /// # Arguments
    /// * `churn` - Churn value (clamped to 0.0..=1.0)
    #[inline]
    pub fn set_current_churn(&mut self, churn: f32) {
        self.current_churn = churn.clamp(0.0, 1.0);
    }

    /// Get churn history with timestamps.
    pub fn get_churn_history(&self) -> Vec<(DateTime<Utc>, f32)> {
        self.churn_history.iter().cloned().collect()
    }

    /// Get number of stored snapshots.
    #[inline]
    pub fn snapshot_count(&self) -> usize {
        self.snapshots.len()
    }

    /// Get most recent snapshot.
    pub fn latest_snapshot(&self) -> Option<&TopicSnapshot> {
        self.snapshots.back()
    }

    /// Compute average churn over specified hours.
    ///
    /// # Arguments
    /// * `hours` - Number of hours to average over
    ///
    /// # Returns
    /// Average churn rate, or 0.0 if no data
    pub fn average_churn(&self, hours: i64) -> f32 {
        let cutoff = Utc::now() - Duration::hours(hours);

        let (sum, count) = self
            .churn_history
            .iter()
            .filter(|(t, _)| *t >= cutoff)
            .fold((0.0f32, 0usize), |(sum, count), (_, churn)| {
                (sum + churn, count + 1)
            });

        if count == 0 {
            return 0.0;
        }

        let avg = sum / count as f32;

        // Guard against NaN per AP-10
        if avg.is_nan() { 0.0 } else { avg }
    }

    /// Check if system is stable (low churn over time).
    ///
    /// Per constitution topic_stability.thresholds: healthy = churn < 0.3
    /// Uses 6-hour average to smooth fluctuations.
    pub fn is_stable(&self) -> bool {
        let avg = self.average_churn(6);
        avg < 0.2 // Conservative threshold for portfolio stability
    }

    /// Get topic count changes since earliest snapshot.
    ///
    /// # Returns
    /// (topics_added, topics_removed) since oldest snapshot
    pub fn topic_count_change(&self) -> (i32, i32) {
        let (Some(oldest), Some(current)) = (self.snapshots.front(), self.snapshots.back()) else {
            return (0, 0);
        };

        if oldest.timestamp == current.timestamp {
            return (0, 0);
        }

        let old_set: HashSet<_> = oldest.topic_ids.iter().collect();
        let cur_set: HashSet<_> = current.topic_ids.iter().collect();

        let added = cur_set.difference(&old_set).count() as i32;
        let removed = old_set.difference(&cur_set).count() as i32;

        (added, removed)
    }

    /// Reset high-entropy tracking (call after dream completes).
    pub fn reset_entropy_tracking(&mut self) {
        self.high_entropy_start = None;
    }

    /// Remove snapshots older than 24 hours.
    fn cleanup_old_snapshots(&mut self) {
        let cutoff = Utc::now() - Duration::hours(SNAPSHOT_RETENTION_HOURS);
        Self::cleanup_before(&mut self.snapshots, |s| s.timestamp, cutoff);
    }

    /// Remove churn history entries older than 24 hours.
    fn cleanup_old_churn_history(&mut self, now: DateTime<Utc>) {
        let cutoff = now - Duration::hours(SNAPSHOT_RETENTION_HOURS);
        Self::cleanup_before(&mut self.churn_history, |(ts, _)| *ts, cutoff);
    }

    /// Generic helper to remove entries with timestamp before cutoff.
    fn cleanup_before<T, F>(deque: &mut VecDeque<T>, get_time: F, cutoff: DateTime<Utc>)
    where
        F: Fn(&T) -> DateTime<Utc>,
    {
        while let Some(entry) = deque.front() {
            if get_time(entry) < cutoff {
                deque.pop_front();
            } else {
                break;
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::clustering::topic::TopicProfile;
    use std::collections::HashMap;

    /// Create test topics with specific IDs.
    fn create_test_topics(ids: &[Uuid]) -> Vec<Topic> {
        ids.iter()
            .map(|&id| {
                let mut topic = Topic::new(
                    TopicProfile::new([0.8, 0.7, 0.6, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]),
                    HashMap::new(),
                    vec![Uuid::new_v4()], // At least one member
                );
                // Override the auto-generated ID
                topic.id = id;
                topic
            })
            .collect()
    }

    /// Create topics with random IDs.
    fn create_random_topics(count: usize) -> Vec<Topic> {
        (0..count)
            .map(|_| {
                Topic::new(
                    TopicProfile::new([0.8, 0.7, 0.6, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]),
                    HashMap::new(),
                    vec![Uuid::new_v4()],
                )
            })
            .collect()
    }

    // ===== TopicSnapshot Tests =====

    #[test]
    fn test_snapshot_creation() {
        println!("=== TEST: test_snapshot_creation ===");
        let topics = create_random_topics(5);

        println!("STATE BEFORE: 5 topics created");

        let snapshot = TopicSnapshot::from_topics(&topics);

        println!("STATE AFTER: snapshot.topic_ids.len() = {}", snapshot.topic_ids.len());
        println!("STATE AFTER: snapshot.total_members = {}", snapshot.total_members);

        assert_eq!(snapshot.topic_ids.len(), 5);
        assert_eq!(snapshot.total_members, 5); // 1 member each

        // SOURCE OF TRUTH: snapshot correctly captures topic state
        for (i, topic) in topics.iter().enumerate() {
            assert_eq!(snapshot.topic_ids[i], topic.id, "Topic ID mismatch at index {}", i);
        }

        println!("[PASS] Snapshot correctly captures topic portfolio\n");
    }

    #[test]
    fn test_snapshot_empty_topics() {
        println!("=== TEST: test_snapshot_empty_topics ===");
        let topics: Vec<Topic> = vec![];

        println!("STATE BEFORE: empty topic list");

        let snapshot = TopicSnapshot::from_topics(&topics);

        println!("STATE AFTER: snapshot.topic_ids.len() = {}", snapshot.topic_ids.len());

        assert!(snapshot.topic_ids.is_empty());
        assert_eq!(snapshot.total_members, 0);

        println!("[PASS] Empty topics handled correctly\n");
    }

    // ===== TopicStabilityTracker Tests =====

    #[test]
    fn test_take_snapshot() {
        println!("=== TEST: test_take_snapshot ===");
        let mut tracker = TopicStabilityTracker::new();
        let topics = create_random_topics(5);

        println!("STATE BEFORE: tracker.snapshot_count() = {}", tracker.snapshot_count());

        tracker.take_snapshot(&topics);

        println!("STATE AFTER: tracker.snapshot_count() = {}", tracker.snapshot_count());

        assert_eq!(tracker.snapshot_count(), 1);

        tracker.take_snapshot(&topics);

        println!("STATE AFTER 2nd snapshot: tracker.snapshot_count() = {}", tracker.snapshot_count());

        assert_eq!(tracker.snapshot_count(), 2);

        println!("[PASS] Snapshots stored correctly\n");
    }

    #[test]
    fn test_churn_calculation_no_change() {
        println!("=== TEST: test_churn_calculation_no_change ===");
        let tracker = TopicStabilityTracker::new();

        let id1 = Uuid::new_v4();
        let id2 = Uuid::new_v4();

        // Same topics in both snapshots
        let old = TopicSnapshot {
            timestamp: Utc::now() - Duration::hours(2),
            topic_ids: vec![id1, id2],
            total_members: 10,
        };
        let current = TopicSnapshot {
            timestamp: Utc::now(),
            topic_ids: vec![id1, id2],
            total_members: 10,
        };

        println!("STATE BEFORE: old={:?}, current={:?}", old.topic_ids, current.topic_ids);

        let churn = tracker.compute_churn(&old, &current);

        println!("STATE AFTER: churn = {}", churn);

        // No change = 0 churn
        assert!(churn.abs() < 0.001, "Expected 0.0 churn, got {}", churn);

        println!("[PASS] No change = 0.0 churn\n");
    }

    #[test]
    fn test_churn_calculation_partial_change() {
        println!("=== TEST: test_churn_calculation_partial_change ===");
        let tracker = TopicStabilityTracker::new();

        let id1 = Uuid::new_v4();
        let id2 = Uuid::new_v4();
        let id3 = Uuid::new_v4();

        let old = TopicSnapshot {
            timestamp: Utc::now() - Duration::hours(2),
            topic_ids: vec![id1, id2],
            total_members: 10,
        };
        // id2 removed, id3 added
        let current = TopicSnapshot {
            timestamp: Utc::now(),
            topic_ids: vec![id1, id3],
            total_members: 10,
        };

        println!("STATE BEFORE:");
        println!("  old topics: [id1, id2]");
        println!("  current topics: [id1, id3]");
        println!("  expected: 1 removed (id2) + 1 added (id3) = 2 changes");
        println!("  union size: 3 (id1, id2, id3)");
        println!("  expected churn: 2/3 = 0.666...");

        let churn = tracker.compute_churn(&old, &current);

        println!("STATE AFTER: computed churn = {}", churn);

        // 1 removed + 1 added = 2 changes, union = 3
        // churn = 2/3 ≈ 0.666
        assert!((churn - (2.0 / 3.0)).abs() < 0.01, "Expected ~0.666, got {}", churn);

        println!("[PASS] Partial change churn computed correctly\n");
    }

    #[test]
    fn test_churn_calculation_complete_change() {
        println!("=== TEST: test_churn_calculation_complete_change ===");
        let tracker = TopicStabilityTracker::new();

        let id1 = Uuid::new_v4();
        let id2 = Uuid::new_v4();
        let id3 = Uuid::new_v4();
        let id4 = Uuid::new_v4();

        let old = TopicSnapshot {
            timestamp: Utc::now() - Duration::hours(2),
            topic_ids: vec![id1, id2],
            total_members: 10,
        };
        // Complete replacement
        let current = TopicSnapshot {
            timestamp: Utc::now(),
            topic_ids: vec![id3, id4],
            total_members: 10,
        };

        println!("STATE BEFORE: complete topic replacement");

        let churn = tracker.compute_churn(&old, &current);

        println!("STATE AFTER: churn = {}", churn);

        // 2 removed + 2 added = 4 changes, union = 4
        // churn = 4/4 = 1.0
        assert!((churn - 1.0).abs() < 0.001, "Expected 1.0 (complete turnover), got {}", churn);

        println!("[PASS] Complete turnover = 1.0 churn\n");
    }

    #[test]
    fn test_churn_calculation_empty_snapshots() {
        println!("=== TEST: test_churn_calculation_empty_snapshots ===");
        let tracker = TopicStabilityTracker::new();

        let old = TopicSnapshot {
            timestamp: Utc::now() - Duration::hours(2),
            topic_ids: vec![],
            total_members: 0,
        };
        let current = TopicSnapshot {
            timestamp: Utc::now(),
            topic_ids: vec![],
            total_members: 0,
        };

        println!("STATE BEFORE: both snapshots empty");

        let churn = tracker.compute_churn(&old, &current);

        println!("STATE AFTER: churn = {}", churn);

        // Empty to empty = 0 churn (no division by zero)
        assert_eq!(churn, 0.0, "Empty snapshots should yield 0.0 churn");
        assert!(!churn.is_nan(), "Churn should not be NaN");

        println!("[PASS] Empty snapshots handled without NaN\n");
    }

    // ===== Dream Trigger Tests =====

    #[test]
    fn test_dream_trigger_high_entropy_and_churn() {
        println!("=== TEST: test_dream_trigger_high_entropy_and_churn ===");
        let mut tracker = TopicStabilityTracker::new();
        tracker.current_churn = 0.6; // > 0.5 threshold

        println!("STATE BEFORE: churn = {}, entropy = 0.8", tracker.current_churn);
        println!("CONDITION: entropy (0.8) > threshold (0.7) AND churn (0.6) > threshold (0.5)");

        let should_trigger = tracker.check_dream_trigger(0.8);

        println!("STATE AFTER: should_trigger = {}", should_trigger);

        assert!(should_trigger, "High entropy + high churn should trigger dream (AP-70)");

        println!("[PASS] Dream triggered on high entropy AND high churn (AP-70)\n");
    }

    #[test]
    fn test_dream_trigger_low_entropy() {
        println!("=== TEST: test_dream_trigger_low_entropy ===");
        let mut tracker = TopicStabilityTracker::new();
        tracker.current_churn = 0.6; // High churn

        println!("STATE BEFORE: churn = {}, entropy = 0.3", tracker.current_churn);
        println!("CONDITION: entropy (0.3) < threshold (0.7), regardless of churn");

        let should_trigger = tracker.check_dream_trigger(0.3);

        println!("STATE AFTER: should_trigger = {}", should_trigger);

        assert!(!should_trigger, "Low entropy should NOT trigger dream even with high churn");

        println!("[PASS] Low entropy prevents dream trigger\n");
    }

    #[test]
    fn test_dream_trigger_high_entropy_low_churn() {
        println!("=== TEST: test_dream_trigger_high_entropy_low_churn ===");
        let mut tracker = TopicStabilityTracker::new();
        tracker.current_churn = 0.2; // Low churn

        println!("STATE BEFORE: churn = {}, entropy = 0.8", tracker.current_churn);
        println!("CONDITION: entropy high but churn low -> need sustained entropy");

        // First call - starts tracking
        let should_trigger_1 = tracker.check_dream_trigger(0.8);

        println!("STATE AFTER first call: should_trigger = {}", should_trigger_1);
        println!("  high_entropy_start set: {}", tracker.high_entropy_start.is_some());

        assert!(!should_trigger_1, "Should not trigger immediately without duration");
        assert!(tracker.high_entropy_start.is_some(), "Should start tracking high entropy");

        println!("[PASS] High entropy + low churn starts duration tracking\n");
    }

    #[test]
    fn test_dream_trigger_sustained_entropy() {
        println!("=== TEST: test_dream_trigger_sustained_entropy ===");
        let mut tracker = TopicStabilityTracker::with_thresholds(0.5, 0.7, 60); // 60 sec for test
        tracker.current_churn = 0.2; // Low churn

        // Simulate that high entropy started 2 minutes ago
        tracker.high_entropy_start = Some(Utc::now() - Duration::seconds(120));

        println!("STATE BEFORE: entropy started 120s ago, threshold = 60s");

        let should_trigger = tracker.check_dream_trigger(0.8);

        println!("STATE AFTER: should_trigger = {}", should_trigger);

        assert!(should_trigger, "Sustained high entropy (120s > 60s) should trigger");

        println!("[PASS] Sustained high entropy triggers dream\n");
    }

    #[test]
    fn test_entropy_tracking_reset_on_low() {
        println!("=== TEST: test_entropy_tracking_reset_on_low ===");
        let mut tracker = TopicStabilityTracker::new();

        // Start high entropy tracking
        tracker.check_dream_trigger(0.8);
        assert!(tracker.high_entropy_start.is_some());

        println!("STATE BEFORE reset: high_entropy_start = {:?}", tracker.high_entropy_start);

        // Entropy drops - should reset tracking
        tracker.check_dream_trigger(0.5);

        println!("STATE AFTER low entropy: high_entropy_start = {:?}", tracker.high_entropy_start);

        assert!(tracker.high_entropy_start.is_none(), "Low entropy should reset tracking");

        println!("[PASS] Low entropy resets duration tracking\n");
    }

    #[test]
    fn test_reset_entropy_tracking() {
        println!("=== TEST: test_reset_entropy_tracking ===");
        let mut tracker = TopicStabilityTracker::new();

        tracker.check_dream_trigger(0.8);
        assert!(tracker.high_entropy_start.is_some());

        println!("STATE BEFORE: high_entropy_start is Some");

        tracker.reset_entropy_tracking();

        println!("STATE AFTER: high_entropy_start = {:?}", tracker.high_entropy_start);

        assert!(tracker.high_entropy_start.is_none());

        println!("[PASS] Manual reset clears entropy tracking\n");
    }

    // ===== History and Stats Tests =====

    #[test]
    fn test_churn_history() {
        println!("=== TEST: test_churn_history ===");
        let mut tracker = TopicStabilityTracker::new();

        // Need at least 2 snapshots with time gap for churn calc
        let id1 = Uuid::new_v4();
        let _topics1 = create_test_topics(&[id1]);

        // Manually add old snapshot
        tracker.snapshots.push_back(TopicSnapshot {
            timestamp: Utc::now() - Duration::hours(2),
            topic_ids: vec![id1],
            total_members: 1,
        });

        // Add different current topics
        let id2 = Uuid::new_v4();
        let topics2 = create_test_topics(&[id2]);
        tracker.take_snapshot(&topics2);

        println!("STATE BEFORE track_churn: history len = {}", tracker.churn_history.len());

        tracker.track_churn();

        let history = tracker.get_churn_history();

        println!("STATE AFTER: history len = {}", history.len());

        assert!(!history.is_empty(), "History should have entries");

        println!("[PASS] Churn history tracked correctly\n");
    }

    #[test]
    fn test_average_churn() {
        println!("=== TEST: test_average_churn ===");
        let mut tracker = TopicStabilityTracker::new();

        // Manually add churn history entries
        let now = Utc::now();
        tracker.churn_history.push_back((now - Duration::hours(1), 0.3));
        tracker.churn_history.push_back((now - Duration::minutes(30), 0.4));
        tracker.churn_history.push_back((now, 0.5));

        println!("STATE BEFORE: 3 churn entries [0.3, 0.4, 0.5]");

        let avg = tracker.average_churn(2);

        println!("STATE AFTER: average_churn(2h) = {}", avg);

        // (0.3 + 0.4 + 0.5) / 3 = 0.4
        assert!((avg - 0.4).abs() < 0.01, "Expected ~0.4, got {}", avg);

        println!("[PASS] Average churn computed correctly\n");
    }

    #[test]
    fn test_is_stable() {
        println!("=== TEST: test_is_stable ===");
        let mut tracker = TopicStabilityTracker::new();

        // No history = stable (no evidence of instability)
        println!("STATE BEFORE: empty history");
        assert!(tracker.is_stable(), "Empty history should be considered stable");

        // Add low churn history
        let now = Utc::now();
        tracker.churn_history.push_back((now - Duration::hours(5), 0.1));
        tracker.churn_history.push_back((now - Duration::hours(4), 0.15));
        tracker.churn_history.push_back((now - Duration::hours(3), 0.1));

        println!("STATE AFTER: low churn entries [0.1, 0.15, 0.1]");

        assert!(tracker.is_stable(), "Low churn should be stable");

        // Add high churn
        tracker.churn_history.push_back((now - Duration::hours(2), 0.6));
        tracker.churn_history.push_back((now - Duration::hours(1), 0.5));
        tracker.churn_history.push_back((now, 0.4));

        println!("STATE AFTER high churn: is_stable = {}", tracker.is_stable());

        assert!(!tracker.is_stable(), "High recent churn should not be stable");

        println!("[PASS] Stability detection works correctly\n");
    }

    #[test]
    fn test_topic_count_change() {
        println!("=== TEST: test_topic_count_change ===");
        let mut tracker = TopicStabilityTracker::new();

        let id1 = Uuid::new_v4();
        let id2 = Uuid::new_v4();
        let id3 = Uuid::new_v4();

        // First snapshot: 2 topics
        tracker.snapshots.push_back(TopicSnapshot {
            timestamp: Utc::now() - Duration::hours(1),
            topic_ids: vec![id1, id2],
            total_members: 10,
        });

        // Second snapshot: id2 removed, id3 added
        tracker.snapshots.push_back(TopicSnapshot {
            timestamp: Utc::now(),
            topic_ids: vec![id1, id3],
            total_members: 10,
        });

        println!("STATE: oldest=[id1, id2], current=[id1, id3]");

        let (added, removed) = tracker.topic_count_change();

        println!("RESULT: added={}, removed={}", added, removed);

        assert_eq!(added, 1, "Should have 1 topic added");
        assert_eq!(removed, 1, "Should have 1 topic removed");

        println!("[PASS] Topic count change tracked correctly\n");
    }

    // ===== Edge Cases =====

    #[test]
    fn test_single_snapshot_no_churn() {
        println!("=== TEST: test_single_snapshot_no_churn ===");
        let mut tracker = TopicStabilityTracker::new();

        let topics = create_random_topics(3);
        tracker.take_snapshot(&topics);

        println!("STATE: only 1 snapshot available");

        let churn = tracker.track_churn();

        println!("RESULT: churn = {}", churn);

        assert_eq!(churn, 0.0, "Single snapshot should yield 0.0 churn");

        println!("[PASS] Single snapshot handled gracefully\n");
    }

    #[test]
    fn test_default_thresholds() {
        println!("=== TEST: test_default_thresholds ===");
        let tracker = TopicStabilityTracker::new();

        println!("Default churn_threshold = {}", tracker.churn_threshold);
        println!("Default entropy_threshold = {}", tracker.entropy_threshold);
        println!("Default entropy_duration_secs = {}", tracker.entropy_duration_secs);

        assert!((tracker.churn_threshold - 0.5).abs() < f32::EPSILON);
        assert!((tracker.entropy_threshold - 0.7).abs() < f32::EPSILON);
        assert_eq!(tracker.entropy_duration_secs, 300);

        println!("[PASS] Default thresholds match constitution\n");
    }

    #[test]
    fn test_custom_thresholds_clamping() {
        println!("=== TEST: test_custom_thresholds_clamping ===");
        let tracker = TopicStabilityTracker::with_thresholds(1.5, -0.5, 100);

        println!("Input: churn=1.5, entropy=-0.5");
        println!("Result: churn={}, entropy={}", tracker.churn_threshold, tracker.entropy_threshold);

        assert!((tracker.churn_threshold - 1.0).abs() < f32::EPSILON, "Churn should clamp to 1.0");
        assert!((tracker.entropy_threshold - 0.0).abs() < f32::EPSILON, "Entropy should clamp to 0.0");

        println!("[PASS] Custom thresholds clamped correctly\n");
    }

    // ===== Boundary Condition Tests =====

    #[test]
    fn test_dream_trigger_boundary_exactly_at_threshold() {
        println!("=== TEST: test_dream_trigger_boundary_exactly_at_threshold ===");
        let mut tracker = TopicStabilityTracker::new();
        tracker.current_churn = 0.5; // Exactly at threshold

        println!("STATE: entropy=0.7, churn=0.5 (exactly at thresholds)");

        let should_trigger = tracker.check_dream_trigger(0.7);

        println!("RESULT: should_trigger = {}", should_trigger);

        // Per AP-70: need entropy > 0.7, not >=
        assert!(!should_trigger, "At threshold (not above) should NOT trigger");

        println!("[PASS] Boundary condition: exactly at threshold does not trigger\n");
    }

    #[test]
    fn test_dream_trigger_just_above_threshold() {
        println!("=== TEST: test_dream_trigger_just_above_threshold ===");
        let mut tracker = TopicStabilityTracker::new();
        tracker.current_churn = 0.51; // Just above threshold

        println!("STATE: entropy=0.71, churn=0.51 (just above thresholds)");

        let should_trigger = tracker.check_dream_trigger(0.71);

        println!("RESULT: should_trigger = {}", should_trigger);

        assert!(should_trigger, "Just above thresholds should trigger");

        println!("[PASS] Just above threshold triggers dream\n");
    }

    #[test]
    fn test_churn_from_empty_to_populated() {
        println!("=== TEST: test_churn_from_empty_to_populated ===");
        let tracker = TopicStabilityTracker::new();

        let id1 = Uuid::new_v4();
        let id2 = Uuid::new_v4();

        let old = TopicSnapshot {
            timestamp: Utc::now() - Duration::hours(1),
            topic_ids: vec![],
            total_members: 0,
        };
        let current = TopicSnapshot {
            timestamp: Utc::now(),
            topic_ids: vec![id1, id2],
            total_members: 10,
        };

        println!("STATE: empty -> 2 topics");

        let churn = tracker.compute_churn(&old, &current);

        println!("RESULT: churn = {}", churn);

        // 2 added, 0 removed, union = 2, churn = 2/2 = 1.0
        assert!((churn - 1.0).abs() < 0.001, "Empty to populated = 1.0 churn");

        println!("[PASS] Empty to populated handled correctly\n");
    }

    #[test]
    fn test_churn_from_populated_to_empty() {
        println!("=== TEST: test_churn_from_populated_to_empty ===");
        let tracker = TopicStabilityTracker::new();

        let id1 = Uuid::new_v4();
        let id2 = Uuid::new_v4();

        let old = TopicSnapshot {
            timestamp: Utc::now() - Duration::hours(1),
            topic_ids: vec![id1, id2],
            total_members: 10,
        };
        let current = TopicSnapshot {
            timestamp: Utc::now(),
            topic_ids: vec![],
            total_members: 0,
        };

        println!("STATE: 2 topics -> empty");

        let churn = tracker.compute_churn(&old, &current);

        println!("RESULT: churn = {}", churn);

        // 0 added, 2 removed, union = 2, churn = 2/2 = 1.0
        assert!((churn - 1.0).abs() < 0.001, "Populated to empty = 1.0 churn");

        println!("[PASS] Populated to empty handled correctly\n");
    }

    #[test]
    fn test_nan_prevention_in_average_churn() {
        println!("=== TEST: test_nan_prevention_in_average_churn ===");
        let tracker = TopicStabilityTracker::new();

        // Empty history
        let avg = tracker.average_churn(6);

        println!("Empty history average = {}", avg);

        assert!(!avg.is_nan(), "Average should not be NaN for empty history");
        assert_eq!(avg, 0.0, "Empty history should return 0.0");

        println!("[PASS] NaN prevention in average_churn\n");
    }

    #[test]
    fn test_latest_snapshot_returns_most_recent() {
        println!("=== TEST: test_latest_snapshot_returns_most_recent ===");
        let mut tracker = TopicStabilityTracker::new();

        let topics1 = create_random_topics(2);
        let topics2 = create_random_topics(5);

        tracker.take_snapshot(&topics1);
        tracker.take_snapshot(&topics2);

        println!("STATE: 2 snapshots taken, latest has 5 topics");

        let latest = tracker.latest_snapshot();

        assert!(latest.is_some());
        assert_eq!(latest.unwrap().topic_ids.len(), 5, "Latest should have 5 topics");

        println!("[PASS] latest_snapshot returns most recent\n");
    }

    #[test]
    fn test_snapshot_count_change_single_snapshot() {
        println!("=== TEST: test_snapshot_count_change_single_snapshot ===");
        let mut tracker = TopicStabilityTracker::new();

        let topics = create_random_topics(3);
        tracker.take_snapshot(&topics);

        println!("STATE: only 1 snapshot");

        let (added, removed) = tracker.topic_count_change();

        println!("RESULT: added={}, removed={}", added, removed);

        assert_eq!(added, 0, "Single snapshot should show 0 added");
        assert_eq!(removed, 0, "Single snapshot should show 0 removed");

        println!("[PASS] Single snapshot returns (0, 0)\n");
    }
}
