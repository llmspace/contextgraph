//! Full State Verification (FSV) tests for TopicStabilityTracker.
//!
//! These tests verify physical state after operations, following the
//! principle of "trust but verify" - we check that the actual stored
//! state matches our expectations.

use context_graph_core::clustering::{
    Topic, TopicProfile, TopicSnapshot, TopicStabilityTracker, DEFAULT_CHURN_THRESHOLD,
    DEFAULT_ENTROPY_DURATION_SECS, DEFAULT_ENTROPY_THRESHOLD, SNAPSHOT_RETENTION_HOURS,
};
use std::collections::HashMap;
use uuid::Uuid;

/// Helper to create a topic with specific ID.
fn create_topic_with_id(id: Uuid) -> Topic {
    let mut topic = Topic::new(
        TopicProfile::new([0.8, 0.7, 0.6, 0.5, 0.5, 0.5, 0.5, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]),
        HashMap::new(),
        vec![Uuid::new_v4()],
    );
    topic.id = id;
    topic
}

// =============================================================================
// FSV Test 1: Snapshot Physical State Verification
// =============================================================================

#[test]
fn fsv_snapshot_captures_correct_topic_ids() {
    println!("\n=== FSV TEST 1: Snapshot Topic ID Capture ===");

    // SETUP: Create 3 topics with known IDs
    let id1 = Uuid::new_v4();
    let id2 = Uuid::new_v4();
    let id3 = Uuid::new_v4();

    let topics = vec![
        create_topic_with_id(id1),
        create_topic_with_id(id2),
        create_topic_with_id(id3),
    ];

    // EXECUTE: Take snapshot
    let mut tracker = TopicStabilityTracker::new();
    println!("STATE BEFORE: snapshot_count = {}", tracker.snapshot_count());

    tracker.take_snapshot(&topics);

    // VERIFY PHYSICAL STATE
    println!("STATE AFTER: snapshot_count = {}", tracker.snapshot_count());

    let latest = tracker.latest_snapshot().expect("Should have snapshot");

    println!("\nPHYSICAL STATE VERIFICATION:");
    println!("  Snapshot topic_ids count: {}", latest.topic_ids.len());
    println!("  Snapshot total_members: {}", latest.total_members);
    println!("  Contains id1 ({}): {}", id1, latest.topic_ids.contains(&id1));
    println!("  Contains id2 ({}): {}", id2, latest.topic_ids.contains(&id2));
    println!("  Contains id3 ({}): {}", id3, latest.topic_ids.contains(&id3));

    assert_eq!(latest.topic_ids.len(), 3, "Must have exactly 3 topic IDs");
    assert_eq!(latest.total_members, 3, "Must have 3 total members (1 per topic)");
    assert!(latest.topic_ids.contains(&id1), "Must contain id1");
    assert!(latest.topic_ids.contains(&id2), "Must contain id2");
    assert!(latest.topic_ids.contains(&id3), "Must contain id3");

    println!("\n[PASS] Snapshot physical state verified");
}

// =============================================================================
// FSV Test 2: Dream Trigger Boundary Conditions (AP-70)
// =============================================================================

#[test]
fn fsv_dream_trigger_boundary_at_threshold() {
    println!("\n=== FSV TEST 2: Dream Trigger At Threshold ===");

    let mut tracker = TopicStabilityTracker::new();

    // Edge case: entropy exactly 0.7, churn exactly 0.5
    // Per AP-70: need entropy > 0.7 AND churn > 0.5
    tracker.set_current_churn(0.5);

    println!("STATE: entropy=0.7 (exactly at threshold), churn=0.5 (exactly at threshold)");
    let result = tracker.check_dream_trigger(0.7);

    println!("RESULT: should_trigger = {}", result);
    println!("EXPECTED: false (must be ABOVE thresholds, not equal)");

    assert!(!result, "At threshold values must NOT trigger dream (need > not >=)");

    println!("\n[PASS] Boundary at threshold verified");
}

#[test]
fn fsv_dream_trigger_boundary_just_above() {
    println!("\n=== FSV TEST 2b: Dream Trigger Just Above Threshold ===");

    let mut tracker = TopicStabilityTracker::new();

    // Just above threshold
    tracker.set_current_churn(0.5001);

    println!("STATE: entropy=0.7001, churn=0.5001 (just above thresholds)");
    let result = tracker.check_dream_trigger(0.7001);

    println!("RESULT: should_trigger = {}", result);
    println!("EXPECTED: true");

    assert!(result, "Just above threshold values must trigger dream");

    println!("\n[PASS] Just above threshold verified");
}

#[test]
fn fsv_dream_trigger_high_entropy_only() {
    println!("\n=== FSV TEST 2c: High Entropy but Low Churn ===");

    let mut tracker = TopicStabilityTracker::new();

    // High entropy but low churn - should NOT immediately trigger
    // (needs sustained duration)
    tracker.set_current_churn(0.2);

    println!("STATE: entropy=0.9 (high), churn=0.2 (low)");
    let result = tracker.check_dream_trigger(0.9);

    println!("RESULT: should_trigger = {}", result);
    println!("EXPECTED: false (need both conditions or sustained entropy)");

    assert!(!result, "High entropy alone should not immediately trigger");

    println!("\n[PASS] High entropy only verified");
}

// =============================================================================
// FSV Test 3: Churn Calculation Edge Cases
// =============================================================================

#[test]
fn fsv_churn_empty_to_empty() {
    println!("\n=== FSV TEST 3a: Churn Empty to Empty ===");

    let empty: Vec<Topic> = vec![];
    let snapshot = TopicSnapshot::from_topics(&empty);

    println!("STATE: Empty topic list");
    println!("PHYSICAL STATE:");
    println!("  snapshot.topic_ids.len() = {}", snapshot.topic_ids.len());
    println!("  snapshot.total_members = {}", snapshot.total_members);

    assert!(snapshot.topic_ids.is_empty(), "Empty list must produce empty snapshot");
    assert_eq!(snapshot.total_members, 0, "Empty list must have 0 members");

    // Verify churn between two empty states
    let tracker = TopicStabilityTracker::new();
    let avg = tracker.average_churn(6);
    println!("  average_churn(6) = {} (no history)", avg);
    assert_eq!(avg, 0.0, "No history must return 0.0 churn");

    println!("\n[PASS] Empty to empty churn verified");
}

#[test]
fn fsv_churn_single_snapshot() {
    println!("\n=== FSV TEST 3b: Single Snapshot Churn ===");

    let mut tracker = TopicStabilityTracker::new();
    let topics = vec![create_topic_with_id(Uuid::new_v4())];

    tracker.take_snapshot(&topics);

    println!("STATE: Single snapshot taken");
    println!("  snapshot_count = {}", tracker.snapshot_count());

    let churn = tracker.track_churn();

    println!("RESULT: track_churn() = {}", churn);
    println!("EXPECTED: 0.0 (need old snapshot from ~1h ago)");

    assert_eq!(churn, 0.0, "Single snapshot must return 0.0 churn");

    println!("\n[PASS] Single snapshot churn verified");
}

// =============================================================================
// FSV Test 4: Constitution Constants Verification
// =============================================================================

#[test]
fn fsv_constitution_constants() {
    println!("\n=== FSV TEST 4: Constitution Constants ===");

    println!("Verifying constants match constitution values:");
    println!(
        "  DEFAULT_CHURN_THRESHOLD = {} (expected 0.5)",
        DEFAULT_CHURN_THRESHOLD
    );
    println!(
        "  DEFAULT_ENTROPY_THRESHOLD = {} (expected 0.7)",
        DEFAULT_ENTROPY_THRESHOLD
    );
    println!(
        "  DEFAULT_ENTROPY_DURATION_SECS = {} (expected 300)",
        DEFAULT_ENTROPY_DURATION_SECS
    );
    println!(
        "  SNAPSHOT_RETENTION_HOURS = {} (expected 24)",
        SNAPSHOT_RETENTION_HOURS
    );

    assert!(
        (DEFAULT_CHURN_THRESHOLD - 0.5).abs() < f32::EPSILON,
        "Churn threshold must be 0.5 per constitution"
    );
    assert!(
        (DEFAULT_ENTROPY_THRESHOLD - 0.7).abs() < f32::EPSILON,
        "Entropy threshold must be 0.7 per constitution"
    );
    assert_eq!(
        DEFAULT_ENTROPY_DURATION_SECS, 300,
        "Entropy duration must be 300s (5 min) per constitution"
    );
    assert_eq!(
        SNAPSHOT_RETENTION_HOURS, 24,
        "Snapshot retention must be 24h per constitution"
    );

    println!("\n[PASS] Constitution constants verified");
}

// =============================================================================
// FSV Test 5: Topic Count Change Tracking
// =============================================================================

#[test]
fn fsv_topic_count_change_no_snapshots() {
    println!("\n=== FSV TEST 5a: Topic Count Change - No Snapshots ===");

    let tracker = TopicStabilityTracker::new();

    println!("STATE: No snapshots taken");
    let (added, removed) = tracker.topic_count_change();

    println!("RESULT: added={}, removed={}", added, removed);
    println!("EXPECTED: (0, 0)");

    assert_eq!((added, removed), (0, 0), "No snapshots must return (0, 0)");

    println!("\n[PASS] No snapshots topic count verified");
}

#[test]
fn fsv_topic_count_change_single_snapshot() {
    println!("\n=== FSV TEST 5b: Topic Count Change - Single Snapshot ===");

    let mut tracker = TopicStabilityTracker::new();
    let topics = vec![
        create_topic_with_id(Uuid::new_v4()),
        create_topic_with_id(Uuid::new_v4()),
    ];
    tracker.take_snapshot(&topics);

    println!("STATE: Single snapshot with 2 topics");
    let (added, removed) = tracker.topic_count_change();

    println!("RESULT: added={}, removed={}", added, removed);
    println!("EXPECTED: (0, 0) - need 2 snapshots for comparison");

    assert_eq!(
        (added, removed),
        (0, 0),
        "Single snapshot must return (0, 0)"
    );

    println!("\n[PASS] Single snapshot topic count verified");
}

// =============================================================================
// FSV Test 6: Stability Detection
// =============================================================================

#[test]
fn fsv_stability_empty_history() {
    println!("\n=== FSV TEST 6a: Stability - Empty History ===");

    let tracker = TopicStabilityTracker::new();

    println!("STATE: No churn history");
    let is_stable = tracker.is_stable();

    println!("RESULT: is_stable = {}", is_stable);
    println!("EXPECTED: true (no evidence of instability)");

    assert!(is_stable, "Empty history should be considered stable");

    println!("\n[PASS] Empty history stability verified");
}

// =============================================================================
// FSV Test 7: Reset Entropy Tracking
// =============================================================================

#[test]
fn fsv_entropy_tracking_reset() {
    println!("\n=== FSV TEST 7: Entropy Tracking Reset ===");

    let mut tracker = TopicStabilityTracker::new();

    // Start tracking
    tracker.check_dream_trigger(0.8);
    println!("STATE AFTER high entropy: tracking started");

    // Reset
    tracker.reset_entropy_tracking();
    println!("STATE AFTER reset: tracking cleared");

    // Verify by checking a low entropy doesn't trigger
    // (if tracking were still active with old start time, it might trigger)
    let result = tracker.check_dream_trigger(0.5);
    println!("RESULT after reset + low entropy: should_trigger = {}", result);

    assert!(!result, "After reset, low entropy should not trigger");

    println!("\n[PASS] Entropy tracking reset verified");
}

// =============================================================================
// FSV Test 8: Average Churn NaN Prevention
// =============================================================================

#[test]
fn fsv_average_churn_nan_prevention() {
    println!("\n=== FSV TEST 8: Average Churn NaN Prevention ===");

    let tracker = TopicStabilityTracker::new();

    println!("STATE: Empty churn history");
    let avg = tracker.average_churn(24);

    println!("RESULT: average_churn(24) = {}", avg);
    println!("VERIFICATION: is_nan = {}, is_finite = {}", avg.is_nan(), avg.is_finite());

    assert!(!avg.is_nan(), "Average must not be NaN");
    assert!(avg.is_finite(), "Average must be finite");
    assert_eq!(avg, 0.0, "Empty history must return 0.0");

    println!("\n[PASS] NaN prevention verified");
}

// =============================================================================
// FSV Test 9: Latest Snapshot Returns Most Recent
// =============================================================================

#[test]
fn fsv_latest_snapshot_order() {
    println!("\n=== FSV TEST 9: Latest Snapshot Order ===");

    let mut tracker = TopicStabilityTracker::new();

    // Take 3 snapshots with different topic counts
    let topics1 = vec![create_topic_with_id(Uuid::new_v4())];
    let topics2 = vec![
        create_topic_with_id(Uuid::new_v4()),
        create_topic_with_id(Uuid::new_v4()),
    ];
    let topics3 = vec![
        create_topic_with_id(Uuid::new_v4()),
        create_topic_with_id(Uuid::new_v4()),
        create_topic_with_id(Uuid::new_v4()),
    ];

    tracker.take_snapshot(&topics1);
    tracker.take_snapshot(&topics2);
    tracker.take_snapshot(&topics3);

    println!("STATE: 3 snapshots taken (1, 2, 3 topics)");
    println!("  snapshot_count = {}", tracker.snapshot_count());

    let latest = tracker.latest_snapshot().expect("Should have latest");

    println!("RESULT: latest.topic_ids.len() = {}", latest.topic_ids.len());
    println!("EXPECTED: 3 (most recent snapshot)");

    assert_eq!(
        latest.topic_ids.len(),
        3,
        "Latest must return most recent snapshot"
    );

    println!("\n[PASS] Latest snapshot order verified");
}

// =============================================================================
// FSV Test 10: Dream Trigger Both Conditions Required
// =============================================================================

#[test]
fn fsv_dream_trigger_both_conditions() {
    println!("\n=== FSV TEST 10: Both Conditions Required ===");

    let mut tracker = TopicStabilityTracker::new();

    // Case 1: High entropy, low churn
    tracker.set_current_churn(0.2);
    let result1 = tracker.check_dream_trigger(0.9);
    println!("High entropy (0.9), low churn (0.2): trigger={}", result1);

    // Case 2: Low entropy, high churn
    tracker.set_current_churn(0.8);
    let result2 = tracker.check_dream_trigger(0.3);
    println!("Low entropy (0.3), high churn (0.8): trigger={}", result2);

    // Case 3: Both high
    tracker.set_current_churn(0.8);
    let result3 = tracker.check_dream_trigger(0.9);
    println!("High entropy (0.9), high churn (0.8): trigger={}", result3);

    assert!(
        !result1,
        "High entropy alone should not trigger (need churn too)"
    );
    assert!(
        !result2,
        "High churn alone should not trigger (need entropy too)"
    );
    assert!(result3, "Both conditions high must trigger");

    println!("\n[PASS] Both conditions requirement verified");
}
