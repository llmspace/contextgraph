# Task Specification: Purpose Vector History Interface

**Task ID:** TASK-IDENTITY-P0-002
**Version:** 2.0.0
**Status:** COMPLETED
**Layer:** Foundation
**Sequence:** 2
**Estimated Complexity:** Low

---

## Metadata

| Field | Value |
|-------|-------|
| Implements | REQ-IDENTITY-010 |
| Depends On | None (foundation task) |
| Blocks | TASK-IDENTITY-P0-003 |
| Priority | P1 - High |
| Constitution Ref | lines 365-392 (self_ego_node, identity_continuity) |

---

## Critical Implementation Notes

> **NO BACKWARDS COMPATIBILITY**: This implementation must fail fast with robust error logging. Do not add compatibility shims or graceful degradation.

> **NO MOCK DATA**: All tests must use real data structures. Verify actual functionality, not mocked behavior.

> **FAIL FAST**: Any invalid state must panic or return `Err` immediately. Do not silently continue.

---

## Codebase Audit (2026-01-11)

### Existing Implementation Analysis

**File:** `crates/context-graph-core/src/gwt/ego_node.rs`

| Symbol | Line | Current State | Notes |
|--------|------|---------------|-------|
| `SelfEgoNode` | 30-43 | ✅ EXISTS | Has `identity_trajectory: Vec<PurposeSnapshot>` |
| `PurposeSnapshot` | 51-58 | ✅ EXISTS | Fields: `vector`, `timestamp`, `context` |
| `record_purpose_snapshot()` | 91-105 | ✅ EXISTS | Uses `Vec::push()` + FIFO eviction at 1000 |
| `get_latest_snapshot()` | 113-115 | ✅ EXISTS | Returns `Option<&PurposeSnapshot>` |
| `get_historical_purpose_vector()` | 108-110 | ✅ EXISTS | Index-based access |
| `IdentityContinuity` | 180-192 | ✅ EXISTS | Already has IC computation |

### Discrepancy: Vec vs VecDeque

**CURRENT**: `SelfEgoNode.identity_trajectory` uses `Vec<PurposeSnapshot>` (line 40)

**ISSUE**: The existing `Vec::remove(0)` at line 101 is O(n), not O(1).

**DECISION**: This task adds `PurposeVectorHistory` as a **new, separate component** that:
1. Uses `VecDeque` for O(1) operations
2. Does NOT modify existing `SelfEgoNode.identity_trajectory`
3. Can be composed into `IdentityContinuityMonitor` (TASK-IDENTITY-P0-003)

---

## Context

Identity continuity calculation requires comparing consecutive purpose vectors:
- **IC = cos(PV_t, PV_{t-1}) × r(t)** (constitution.yaml line 369)

The existing `SelfEgoNode.identity_trajectory` stores history but lacks:
1. O(1) access to current and previous PV
2. Dedicated trait for abstraction (testability)
3. Explicit "first vector" edge case handling

This task creates `PurposeVectorHistory` as a standalone component for use in `IdentityContinuityMonitor`.

---

## Input Context Files

| File | Purpose | Lines |
|------|---------|-------|
| `crates/context-graph-core/src/gwt/ego_node.rs` | Existing types | 30-58 (SelfEgoNode, PurposeSnapshot) |
| `docs2/constitution.yaml` | IC formula, thresholds | 365-392 |
| `specs/functional/SPEC-IDENTITY-001.md` | Requirements | REQ-IDENTITY-010 |

---

## Prerequisites

- [x] Rust workspace compiles: `cargo build -p context-graph-core`
- [x] `PurposeSnapshot` type exists in `ego_node.rs` (line 51)
- [x] `chrono::Utc` available for timestamps
- [x] `serde::{Serialize, Deserialize}` available

---

## Scope

### In Scope

1. Create `PurposeVectorHistory` struct with `VecDeque<PurposeSnapshot>`
2. Implement `PurposeVectorHistoryProvider` trait
3. Add `MAX_PV_HISTORY_SIZE` constant (1000)
4. Add comprehensive unit tests with FSV

### Out of Scope

- IC computation (TASK-IDENTITY-P0-003)
- Modifying existing `SelfEgoNode.identity_trajectory`
- RocksDB persistence (future task)
- Integration with `SelfAwarenessLoop`

---

## Definition of Done

### Source of Truth

**Primary:** `PurposeVectorHistory` struct in `ego_node.rs`
**Verification:** Unit tests with real `PurposeSnapshot` instances (no mocks)

### Exact Signatures Required

```rust
// File: crates/context-graph-core/src/gwt/ego_node.rs
// Location: After PurposeSnapshot (line ~59)

use std::collections::VecDeque;

/// Maximum purpose vector history size per constitution
/// Reference: constitution.yaml line 390 (identity_trajectory: 1000)
pub const MAX_PV_HISTORY_SIZE: usize = 1000;

/// Manages purpose vector history for identity continuity calculation
///
/// Provides O(1) access to current and previous purpose vectors,
/// handling the edge case of first vector (no previous).
///
/// # Constitution Reference
/// - self_ego_node.identity_trajectory: max 1000 snapshots
/// - IC = cos(PV_t, PV_{t-1}) × r(t) requires consecutive PV access
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
    max_size: usize,
}

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
    /// # Panics
    /// Never panics. max_size of 0 means no eviction limit.
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
```

### Constraints (MUST enforce)

| # | Constraint | Verification |
|---|------------|--------------|
| C1 | `VecDeque` for O(1) push/pop | Code review |
| C2 | `push()` returns previous PV | Unit test |
| C3 | `push()` evicts oldest at capacity | Unit test with max_size=3 |
| C4 | `current()` returns `None` for empty | Unit test |
| C5 | `previous()` returns `None` for len < 2 | Unit test |
| C6 | `is_first_vector()` true IFF len == 1 | Unit test for 0, 1, 2 |
| C7 | NO panics from any operation | Fuzz test / code review |
| C8 | Serialization roundtrip preserves state | bincode + JSON test |

---

## Full State Verification (FSV)

### FSV-1: Push and Retrieve

**Source of Truth:** `PurposeVectorHistory.history` field

**Before State:**
```rust
let mut history = PurposeVectorHistory::new();
assert!(history.is_empty());
assert!(history.current().is_none());
```

**Execute:**
```rust
let prev = history.push([0.5; 13], "Test context");
```

**After State Verification:**
```rust
assert!(prev.is_none()); // No previous for first push
assert_eq!(history.len(), 1);
assert!(history.is_first_vector());
assert!(history.current().is_some());
assert_eq!(*history.current().unwrap(), [0.5; 13]);
assert!(history.previous().is_none());
```

**Evidence:** `history.history().len() == 1`

---

### FSV-2: FIFO Eviction

**Source of Truth:** `PurposeVectorHistory.history` after eviction

**Before State:**
```rust
let mut history = PurposeVectorHistory::with_max_size(3);
history.push([0.1; 13], "1");
history.push([0.2; 13], "2");
history.push([0.3; 13], "3");
assert_eq!(history.len(), 3);
// Oldest is [0.1; 13]
```

**Execute:**
```rust
history.push([0.4; 13], "4");
```

**After State Verification:**
```rust
assert_eq!(history.len(), 3); // Still 3, not 4
// Check oldest was evicted
let all_vectors: Vec<_> = history.history().iter().map(|s| s.vector[0]).collect();
assert!(!all_vectors.contains(&0.1)); // 0.1 was evicted
assert_eq!(*history.current().unwrap(), [0.4; 13]);
assert_eq!(*history.previous().unwrap(), [0.3; 13]);
```

**Evidence:** `history.history().front().unwrap().vector[0] == 0.2` (not 0.1)

---

### FSV-3: Serialization Roundtrip

**Source of Truth:** Deserialized struct fields

**Before State:**
```rust
let mut original = PurposeVectorHistory::with_max_size(100);
original.push([0.8; 13], "Context A");
original.push([0.9; 13], "Context B");
```

**Execute:**
```rust
let serialized = bincode::serialize(&original).expect("serialize must not fail");
let restored: PurposeVectorHistory = bincode::deserialize(&serialized).expect("deserialize must not fail");
```

**After State Verification:**
```rust
assert_eq!(restored.len(), original.len());
assert_eq!(restored.current(), original.current());
assert_eq!(restored.previous(), original.previous());
// Verify max_size preserved
assert_eq!(restored.max_size, original.max_size);
```

**Evidence:** Field-by-field equality check

---

## Edge Case Testing (3 Required)

### EC-1: Empty History

**Scenario:** All accessors on empty history

```rust
#[test]
fn test_edge_case_empty_history() {
    let history = PurposeVectorHistory::new();

    // BEFORE: Empty state
    assert!(history.is_empty());
    assert_eq!(history.len(), 0);

    // VERIFY: All accessors handle gracefully (no panic)
    assert!(history.current().is_none());
    assert!(history.previous().is_none());
    assert!(history.current_and_previous().is_none());
    assert!(!history.is_first_vector()); // Empty is NOT "first vector"

    // EVIDENCE: No panic, all return None/false appropriately
}
```

### EC-2: Single Entry (First Vector)

**Scenario:** Exactly one entry - the "first vector" case

```rust
#[test]
fn test_edge_case_first_vector() {
    let mut history = PurposeVectorHistory::new();

    // BEFORE
    assert!(history.is_empty());

    // EXECUTE
    let prev = history.push([0.77; 13], "First ever");

    // AFTER
    assert!(prev.is_none()); // No previous
    assert!(history.is_first_vector()); // EXACTLY true
    assert_eq!(history.len(), 1);

    // current_and_previous returns (current, None)
    let (curr, prev_ref) = history.current_and_previous().unwrap();
    assert_eq!(*curr, [0.77; 13]);
    assert!(prev_ref.is_none());

    // EVIDENCE: is_first_vector() == true, previous() == None
}
```

### EC-3: Zero Max Size

**Scenario:** max_size = 0 means unlimited

```rust
#[test]
fn test_edge_case_zero_max_size() {
    let mut history = PurposeVectorHistory::with_max_size(0);

    // BEFORE: Empty, unlimited capacity
    assert!(history.is_empty());

    // EXECUTE: Push many entries
    for i in 0..100 {
        history.push([i as f32 * 0.01; 13], format!("Entry {}", i));
    }

    // AFTER: All 100 should exist (no eviction with max_size=0)
    assert_eq!(history.len(), 100);

    // EVIDENCE: len() == 100, no eviction occurred
}
```

---

## Manual Verification Checklist

After implementation, manually verify:

- [ ] `cargo build -p context-graph-core` succeeds
- [ ] `cargo test -p context-graph-core purpose_vector_history` - all tests pass
- [ ] `cargo clippy -p context-graph-core -- -D warnings` - no warnings
- [ ] In test output, FSV tests print BEFORE/AFTER state
- [ ] `history.history()` accessor provides read access to internal VecDeque
- [ ] No `unwrap()` calls in non-test code

---

## Files to Modify

| File | Changes | Location |
|------|---------|----------|
| `crates/context-graph-core/src/gwt/ego_node.rs` | Add import `use std::collections::VecDeque;` | After line 21 |
| `crates/context-graph-core/src/gwt/ego_node.rs` | Add `MAX_PV_HISTORY_SIZE` | After imports |
| `crates/context-graph-core/src/gwt/ego_node.rs` | Add `PurposeVectorHistory` struct | After `PurposeSnapshot` (line ~59) |
| `crates/context-graph-core/src/gwt/ego_node.rs` | Add `PurposeVectorHistoryProvider` trait | After struct |
| `crates/context-graph-core/src/gwt/ego_node.rs` | Add impl blocks | After trait |
| `crates/context-graph-core/src/gwt/ego_node.rs` | Add test module `purpose_vector_history_tests` | In `#[cfg(test)] mod tests` |

---

## Test Cases (Real Data, No Mocks)

```rust
#[cfg(test)]
mod purpose_vector_history_tests {
    use super::*;

    /// Helper: Create a purpose vector with uniform values
    fn uniform_pv(val: f32) -> [f32; 13] {
        [val; 13]
    }

    /// Helper: Create a purpose vector with specific pattern
    fn pattern_pv(base: f32) -> [f32; 13] {
        [
            base, base + 0.05, base + 0.1, base - 0.05, base,
            base + 0.02, base - 0.03, base + 0.08, base - 0.01,
            base + 0.04, base - 0.02, base + 0.06, base - 0.04,
        ]
    }

    // =========================================================================
    // FSV Tests (Full State Verification)
    // =========================================================================

    #[test]
    fn fsv_push_and_retrieve() {
        println!("=== FSV-1: Push and Retrieve ===");

        // BEFORE
        let mut history = PurposeVectorHistory::new();
        println!("BEFORE: is_empty={}, len={}", history.is_empty(), history.len());
        assert!(history.is_empty());
        assert!(history.current().is_none());

        // EXECUTE
        let prev = history.push(uniform_pv(0.5), "Test context");

        // AFTER
        println!("AFTER: is_empty={}, len={}, is_first_vector={}",
                 history.is_empty(), history.len(), history.is_first_vector());
        assert!(prev.is_none());
        assert_eq!(history.len(), 1);
        assert!(history.is_first_vector());
        assert!(history.current().is_some());
        assert_eq!(*history.current().unwrap(), uniform_pv(0.5));
        assert!(history.previous().is_none());

        // EVIDENCE
        println!("EVIDENCE: history.history().len() = {}", history.history().len());
        assert_eq!(history.history().len(), 1);
    }

    #[test]
    fn fsv_fifo_eviction() {
        println!("=== FSV-2: FIFO Eviction ===");

        // BEFORE
        let mut history = PurposeVectorHistory::with_max_size(3);
        history.push(uniform_pv(0.1), "1");
        history.push(uniform_pv(0.2), "2");
        history.push(uniform_pv(0.3), "3");
        println!("BEFORE: len={}", history.len());
        assert_eq!(history.len(), 3);

        // EXECUTE
        history.push(uniform_pv(0.4), "4");

        // AFTER
        println!("AFTER: len={}", history.len());
        assert_eq!(history.len(), 3); // Still 3

        // Verify oldest was evicted
        let oldest_val = history.history().front().unwrap().vector[0];
        println!("EVIDENCE: oldest value = {} (should be 0.2, not 0.1)", oldest_val);
        assert!((oldest_val - 0.2).abs() < 1e-6);
        assert_eq!(*history.current().unwrap(), uniform_pv(0.4));
        assert_eq!(*history.previous().unwrap(), uniform_pv(0.3));
    }

    #[test]
    fn fsv_serialization_roundtrip() {
        println!("=== FSV-3: Serialization Roundtrip ===");

        // BEFORE
        let mut original = PurposeVectorHistory::with_max_size(100);
        original.push(pattern_pv(0.8), "Context A");
        original.push(pattern_pv(0.9), "Context B");
        println!("BEFORE: len={}, max_size={}", original.len(), original.max_size);

        // EXECUTE
        let serialized = bincode::serialize(&original).expect("serialize must not fail");
        let restored: PurposeVectorHistory = bincode::deserialize(&serialized)
            .expect("deserialize must not fail");

        // AFTER
        println!("AFTER: restored.len={}, restored.max_size={}",
                 restored.len(), restored.max_size);
        assert_eq!(restored.len(), original.len());
        assert_eq!(restored.current(), original.current());
        assert_eq!(restored.previous(), original.previous());
        assert_eq!(restored.max_size, original.max_size);

        // EVIDENCE
        println!("EVIDENCE: All fields match original");
    }

    // =========================================================================
    // Edge Case Tests
    // =========================================================================

    #[test]
    fn test_edge_case_empty_history() {
        println!("=== EDGE CASE: Empty History ===");

        let history = PurposeVectorHistory::new();

        assert!(history.is_empty());
        assert_eq!(history.len(), 0);
        assert!(history.current().is_none());
        assert!(history.previous().is_none());
        assert!(history.current_and_previous().is_none());
        assert!(!history.is_first_vector());

        println!("EVIDENCE: All accessors return None/false, no panic");
    }

    #[test]
    fn test_edge_case_first_vector() {
        println!("=== EDGE CASE: First Vector ===");

        let mut history = PurposeVectorHistory::new();
        let prev = history.push(uniform_pv(0.77), "First ever");

        assert!(prev.is_none());
        assert!(history.is_first_vector());
        assert_eq!(history.len(), 1);

        let (curr, prev_ref) = history.current_and_previous().unwrap();
        assert_eq!(*curr, uniform_pv(0.77));
        assert!(prev_ref.is_none());

        println!("EVIDENCE: is_first_vector()=true, previous()=None");
    }

    #[test]
    fn test_edge_case_zero_max_size() {
        println!("=== EDGE CASE: Zero Max Size (Unlimited) ===");

        let mut history = PurposeVectorHistory::with_max_size(0);

        for i in 0..100 {
            history.push(uniform_pv(i as f32 * 0.01), format!("Entry {}", i));
        }

        assert_eq!(history.len(), 100);
        println!("EVIDENCE: len()=100, no eviction with max_size=0");
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
    fn test_json_serialization() {
        let mut history = PurposeVectorHistory::new();
        history.push(uniform_pv(0.75), "JSON test");

        let json = serde_json::to_string(&history).expect("JSON serialize");
        let restored: PurposeVectorHistory = serde_json::from_str(&json).expect("JSON deserialize");

        assert_eq!(restored.len(), history.len());
        assert_eq!(restored.current(), history.current());
    }

    #[test]
    fn test_default_trait() {
        let history = PurposeVectorHistory::default();
        assert!(history.is_empty());
        assert_eq!(history.max_size, MAX_PV_HISTORY_SIZE);
    }

    #[test]
    fn test_context_preserved_in_snapshot() {
        let mut history = PurposeVectorHistory::new();
        history.push(uniform_pv(0.5), "Important context");

        let snapshot = history.history().back().unwrap();
        assert_eq!(snapshot.context, "Important context");
    }

    #[test]
    fn test_timestamp_is_recent() {
        let before = Utc::now();

        let mut history = PurposeVectorHistory::new();
        history.push(uniform_pv(0.5), "Timestamp test");

        let after = Utc::now();

        let snapshot = history.history().back().unwrap();
        assert!(snapshot.timestamp >= before);
        assert!(snapshot.timestamp <= after);
    }
}
```

---

## Verification Commands

```bash
# Build check
cargo build -p context-graph-core

# Run specific tests
cargo test -p context-graph-core purpose_vector_history -- --nocapture

# Run FSV tests specifically
cargo test -p context-graph-core fsv_ -- --nocapture

# Run edge case tests
cargo test -p context-graph-core test_edge_case -- --nocapture

# Clippy (fail on warnings)
cargo clippy -p context-graph-core -- -D warnings

# Check documentation
cargo doc -p context-graph-core --no-deps
```

---

## Error Handling Requirements

1. **No unwrap() in production code** - Use `Option` returns
2. **No panic paths** - All edge cases return `None` or empty
3. **Fail fast principle** - Invalid inputs should return errors, not silent degradation
4. **Logging** - Use `tracing::debug!` for significant operations

---

## Traceability

| Requirement | Task Section | Verification |
|-------------|--------------|--------------|
| REQ-IDENTITY-010 | PurposeVectorHistory struct | FSV-1, FSV-2 |
| EC-IDENTITY-01 (first vector) | is_first_vector() | test_edge_case_first_vector |
| constitution.yaml:390 (1000 limit) | MAX_PV_HISTORY_SIZE | fsv_fifo_eviction |
| SPEC-IDENTITY-001 Section 5.1 | All trait methods | All unit tests |

---

## Revision History

| Version | Date | Author | Changes |
|---------|------|--------|---------|
| 1.0.0 | 2026-01-11 | Claude Opus 4.5 | Initial task specification |
| 2.0.0 | 2026-01-11 | Claude Opus 4.5 | Full rewrite: Added FSV requirements, edge cases, codebase audit, corrected discrepancies |
