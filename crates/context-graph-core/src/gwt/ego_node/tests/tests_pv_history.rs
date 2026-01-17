//! Tests for PurposeVectorHistory
//!
//! Tests FIFO eviction, serialization, and edge cases.

use chrono::Utc;

use super::super::{PurposeVectorHistory, PurposeVectorHistoryProvider, MAX_PV_HISTORY_SIZE};

/// Helper: Create a purpose vector with uniform values
fn uniform_pv(val: f32) -> [f32; 13] {
    [val; 13]
}

/// Helper: Create a purpose vector with specific pattern
fn pattern_pv(base: f32) -> [f32; 13] {
    [
        base,
        base + 0.05,
        base + 0.1,
        base - 0.05,
        base,
        base + 0.02,
        base - 0.03,
        base + 0.08,
        base - 0.01,
        base + 0.04,
        base - 0.02,
        base + 0.06,
        base - 0.04,
    ]
}

// =========================================================================
// FSV Tests (Full State Verification)
// =========================================================================

#[test]
fn fsv_push_and_retrieve() {
    // BEFORE
    let mut history = PurposeVectorHistory::new();
    assert!(history.is_empty());
    assert!(history.current().is_none());

    // EXECUTE
    let prev = history.push(uniform_pv(0.5), "Test context");

    // AFTER
    assert!(prev.is_none());
    assert_eq!(history.len(), 1);
    assert!(history.is_first_vector());
    assert!(history.current().is_some());
    assert_eq!(*history.current().unwrap(), uniform_pv(0.5));
    assert!(history.previous().is_none());

    // EVIDENCE
    assert_eq!(history.history().len(), 1);
}

#[test]
fn fsv_fifo_eviction() {
    // BEFORE
    let mut history = PurposeVectorHistory::with_max_size(3);
    history.push(uniform_pv(0.1), "1");
    history.push(uniform_pv(0.2), "2");
    history.push(uniform_pv(0.3), "3");
    assert_eq!(history.len(), 3);

    // EXECUTE
    history.push(uniform_pv(0.4), "4");

    // AFTER
    assert_eq!(history.len(), 3); // Still 3

    // Verify oldest was evicted
    let oldest_val = history.history().front().unwrap().vector[0];
    assert!((oldest_val - 0.2).abs() < 1e-6);
    assert_eq!(*history.current().unwrap(), uniform_pv(0.4));
    assert_eq!(*history.previous().unwrap(), uniform_pv(0.3));
}

#[test]
fn fsv_serialization_roundtrip() {
    // BEFORE
    let mut original = PurposeVectorHistory::with_max_size(100);
    original.push(pattern_pv(0.8), "Context A");
    original.push(pattern_pv(0.9), "Context B");

    // EXECUTE
    let serialized = bincode::serialize(&original).expect("serialize must not fail");
    let restored: PurposeVectorHistory =
        bincode::deserialize(&serialized).expect("deserialize must not fail");

    // AFTER
    assert_eq!(restored.len(), original.len());
    assert_eq!(restored.current(), original.current());
    assert_eq!(restored.previous(), original.previous());
    assert_eq!(restored.max_size, original.max_size);
}

// =========================================================================
// Edge Case Tests
// =========================================================================

#[test]
fn test_edge_case_empty_history() {
    let history = PurposeVectorHistory::new();

    assert!(history.is_empty());
    assert_eq!(history.len(), 0);
    assert!(history.current().is_none());
    assert!(history.previous().is_none());
    assert!(history.current_and_previous().is_none());
    assert!(!history.is_first_vector());
}

#[test]
fn test_edge_case_first_vector_pv_history() {
    let mut history = PurposeVectorHistory::new();
    let prev = history.push(uniform_pv(0.77), "First ever");

    assert!(prev.is_none());
    assert!(history.is_first_vector());
    assert_eq!(history.len(), 1);

    let (curr, prev_ref) = history.current_and_previous().unwrap();
    assert_eq!(*curr, uniform_pv(0.77));
    assert!(prev_ref.is_none());
}

#[test]
fn test_edge_case_zero_max_size() {
    let mut history = PurposeVectorHistory::with_max_size(0);

    for i in 0..100 {
        history.push(uniform_pv(i as f32 * 0.01), format!("Entry {}", i));
    }

    assert_eq!(history.len(), 100);
}

// =========================================================================
// Core Functionality Tests
// =========================================================================

#[test]
fn test_new_creates_empty_history() {
    let history = PurposeVectorHistory::new();
    assert!(history.is_empty());
    assert_eq!(history.len(), 0);
    assert_eq!(history.max_size, MAX_PV_HISTORY_SIZE);
}

#[test]
fn test_push_returns_previous() {
    let mut history = PurposeVectorHistory::new();

    // First push returns None
    let prev1 = history.push(uniform_pv(0.5), "First");
    assert!(prev1.is_none());

    // Second push returns first
    let prev2 = history.push(uniform_pv(0.7), "Second");
    assert_eq!(prev2.unwrap(), uniform_pv(0.5));

    // Third push returns second
    let prev3 = history.push(uniform_pv(0.9), "Third");
    assert_eq!(prev3.unwrap(), uniform_pv(0.7));
}

#[test]
fn test_current_and_previous_all_states() {
    let mut history = PurposeVectorHistory::new();

    // Empty
    assert!(history.current_and_previous().is_none());

    // One entry
    history.push(uniform_pv(0.5), "1");
    let result = history.current_and_previous().unwrap();
    assert_eq!(*result.0, uniform_pv(0.5));
    assert!(result.1.is_none());

    // Two entries
    history.push(uniform_pv(0.7), "2");
    let result = history.current_and_previous().unwrap();
    assert_eq!(*result.0, uniform_pv(0.7));
    assert_eq!(*result.1.unwrap(), uniform_pv(0.5));
}

#[test]
fn test_is_first_vector_transitions() {
    let mut history = PurposeVectorHistory::new();

    // Empty: NOT first vector
    assert!(!history.is_first_vector());

    // One entry: IS first vector
    history.push(uniform_pv(0.5), "1");
    assert!(history.is_first_vector());

    // Two entries: NOT first vector
    history.push(uniform_pv(0.6), "2");
    assert!(!history.is_first_vector());

    // Many entries: NOT first vector
    history.push(uniform_pv(0.7), "3");
    assert!(!history.is_first_vector());
}

#[test]
fn test_json_serialization_pv_history() {
    let mut history = PurposeVectorHistory::new();
    history.push(uniform_pv(0.75), "JSON test");

    let json = serde_json::to_string(&history).expect("JSON serialize");
    let restored: PurposeVectorHistory = serde_json::from_str(&json).expect("JSON deserialize");

    assert_eq!(restored.len(), history.len());
    assert_eq!(restored.current(), history.current());
}

#[test]
fn test_default_trait_pv_history() {
    let history = PurposeVectorHistory::default();
    assert!(history.is_empty());
    assert_eq!(history.max_size, MAX_PV_HISTORY_SIZE);
}

#[test]
fn test_context_preserved_in_snapshot_pv_history() {
    let mut history = PurposeVectorHistory::new();
    history.push(uniform_pv(0.5), "Important context");

    let snapshot = history.history().back().unwrap();
    assert_eq!(snapshot.context, "Important context");
}

#[test]
fn test_timestamp_is_recent_pv_history() {
    let before = Utc::now();

    let mut history = PurposeVectorHistory::new();
    history.push(uniform_pv(0.5), "Timestamp test");

    let after = Utc::now();

    let snapshot = history.history().back().unwrap();
    assert!(snapshot.timestamp >= before);
    assert!(snapshot.timestamp <= after);
}

// =========================================================================
// Additional PurposeVectorHistory Tests
// =========================================================================

#[test]
fn test_multiple_evictions() {
    let mut history = PurposeVectorHistory::with_max_size(3);

    // Push 10 items, only last 3 should remain
    for i in 0..10 {
        history.push(uniform_pv(i as f32 * 0.1), format!("Entry {}", i));
    }

    assert_eq!(history.len(), 3);

    // Verify the last 3 entries (7, 8, 9)
    let all_vals: Vec<f32> = history.history().iter().map(|s| s.vector[0]).collect();
    assert!(
        (all_vals[0] - 0.7).abs() < 1e-5,
        "Expected 0.7, got {}",
        all_vals[0]
    );
    assert!(
        (all_vals[1] - 0.8).abs() < 1e-5,
        "Expected 0.8, got {}",
        all_vals[1]
    );
    assert!(
        (all_vals[2] - 0.9).abs() < 1e-5,
        "Expected 0.9, got {}",
        all_vals[2]
    );
}

#[test]
fn test_history_accessor_returns_readonly_reference() {
    let mut history = PurposeVectorHistory::new();
    history.push(uniform_pv(0.5), "Test");
    history.push(uniform_pv(0.6), "Test 2");

    // history() should return a read-only reference
    let deque = history.history();
    assert_eq!(deque.len(), 2);
    assert!((deque.front().unwrap().vector[0] - 0.5).abs() < 1e-6);
    assert!((deque.back().unwrap().vector[0] - 0.6).abs() < 1e-6);
}

#[test]
fn test_with_max_size_different_capacities() {
    // Small capacity
    let h1 = PurposeVectorHistory::with_max_size(5);
    assert_eq!(h1.max_size, 5);

    // Large capacity (should still work)
    let h2 = PurposeVectorHistory::with_max_size(10000);
    assert_eq!(h2.max_size, 10000);

    // Default capacity
    let h3 = PurposeVectorHistory::new();
    assert_eq!(h3.max_size, MAX_PV_HISTORY_SIZE);
}
