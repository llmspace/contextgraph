# Task Specification: Meta-UTL Core Types and Accuracy History Interface

**Task ID:** TASK-METAUTL-P0-001
**Version:** 2.0.0
**Status:** Partially Implemented
**Layer:** Foundation (Layer 1)
**Sequence:** 1
**Priority:** P0 (Critical)
**Estimated Complexity:** Medium

---

## 1. Metadata

### 1.1 Implements

| Requirement ID | Description | Current Status |
|----------------|-------------|----------------|
| REQ-METAUTL-001 | Rolling accuracy history of at least 100 predictions | ✅ Implemented in `MetaUtlTracker` |
| REQ-METAUTL-002 | Track accuracy per embedder (E1-E13) separately | ✅ Implemented via `embedder_accuracy` array |
| REQ-METAUTL-006 | Lambda weights SHALL always sum to 1.0 | ✅ Enforced in `update_weights()` |
| REQ-METAUTL-007 | Lambda weights SHALL be clamped to [0.1, 0.9] | ⚠️ NOT YET IMPLEMENTED - weights normalize to sum=1.0 but no clamping |

### 1.2 Dependencies

| Task ID | Description | Status |
|---------|-------------|--------|
| None | This is the foundation task | N/A |

### 1.3 Blocked By

None - this is the first task in the sequence.

---

## 2. Current Implementation State

> **CRITICAL: Review this section before implementation to avoid duplication**

### 2.1 Existing Implementations

The codebase already has significant Meta-UTL implementation:

| Component | Location | Status |
|-----------|----------|--------|
| `MetaUtlTracker` | `crates/context-graph-mcp/src/handlers/core.rs:60-200` | ✅ Implemented |
| `StoredPrediction` | `crates/context-graph-mcp/src/handlers/core.rs:44-53` | ✅ Implemented |
| `PredictionType` | `crates/context-graph-mcp/src/handlers/core.rs:36-42` | ✅ Implemented |
| MCP Handler: `meta_utl/learning_trajectory` | `crates/context-graph-mcp/src/handlers/utl.rs` | ✅ Implemented |
| MCP Handler: `meta_utl/health_metrics` | `crates/context-graph-mcp/src/handlers/utl.rs` | ✅ Implemented |
| MCP Handler: `meta_utl/predict_storage` | `crates/context-graph-mcp/src/handlers/utl.rs` | ✅ Implemented |
| MCP Handler: `meta_utl/predict_retrieval` | `crates/context-graph-mcp/src/handlers/utl.rs` | ✅ Implemented |
| MCP Handler: `meta_utl/validate_prediction` | `crates/context-graph-mcp/src/handlers/utl.rs` | ✅ Implemented |
| MCP Handler: `meta_utl/optimized_weights` | `crates/context-graph-mcp/src/handlers/utl.rs` | ✅ Implemented |
| FSV Tests | `crates/context-graph-mcp/src/handlers/tests/full_state_verification_meta_utl.rs` | ✅ Implemented |

### 2.2 Existing MetaUtlTracker Structure

```rust
// Location: crates/context-graph-mcp/src/handlers/core.rs:60-77
pub struct MetaUtlTracker {
    pub pending_predictions: HashMap<Uuid, StoredPrediction>,
    pub embedder_accuracy: [[f32; 100]; NUM_EMBEDDERS],  // 13 embedders × 100 samples
    pub accuracy_indices: [usize; NUM_EMBEDDERS],
    pub accuracy_counts: [usize; NUM_EMBEDDERS],
    pub current_weights: [f32; NUM_EMBEDDERS],           // Sum to 1.0
    pub prediction_count: usize,
    pub validation_count: usize,
    pub last_weight_update: Option<Instant>,
}
```

### 2.3 What Still Needs Implementation

| Component | Description | Priority |
|-----------|-------------|----------|
| Lambda weight clamping | REQ-METAUTL-007: Clamp to [0.1, 0.9] | P0 |
| `Domain` enum | Domain-specific accuracy tracking | P1 |
| `MetaLearningEvent` | Event logging struct | P1 |
| `SelfCorrectionConfig` | Configuration struct | P1 |
| Bayesian escalation trigger | When accuracy < 0.7 for 10 cycles | P0 |
| Consecutive low tracking | Track consecutive low accuracy cycles | P0 |

---

## 3. Context

### 3.1 Constitution Reference

From `docs2/constitution.yaml`:

```yaml
meta_utl:
  self_correction:
    enabled: true
    threshold: 0.2  # Prediction error threshold
    max_consecutive_failures: 10
    escalation_strategy: "bayesian_optimization"
```

### 3.2 File Structure

The current codebase organizes Meta-UTL in the MCP handlers crate, NOT in the UTL crate:

```
crates/
├── context-graph-mcp/
│   └── src/
│       └── handlers/
│           ├── core.rs         # MetaUtlTracker, StoredPrediction, PredictionType
│           ├── utl.rs          # 6 meta_utl/* MCP handlers
│           └── tests/
│               └── full_state_verification_meta_utl.rs
├── context-graph-utl/
│   └── src/
│       ├── lifecycle/
│       │   └── lambda.rs       # LifecycleLambdaWeights (stage-based, NOT self-correcting)
│       └── lib.rs              # NO meta/ module currently
└── context-graph-core/
    └── src/
        └── johari/
            └── manager.rs      # NUM_EMBEDDERS = 13 constant
```

---

## 4. Input Context Files (MUST READ)

| File | Purpose | Read Priority |
|------|---------|---------------|
| `crates/context-graph-mcp/src/handlers/core.rs:36-200` | **Existing MetaUtlTracker** | P0 - Read First |
| `crates/context-graph-mcp/src/handlers/utl.rs` | Existing MCP handlers | P0 |
| `crates/context-graph-mcp/src/handlers/tests/full_state_verification_meta_utl.rs` | Existing FSV tests | P0 |
| `docs2/constitution.yaml` (lines 200-220) | Authoritative constraints | P0 |
| `specs/functional/SPEC-METAUTL-001.md` | Full functional specification | P1 |
| `crates/context-graph-core/src/johari/manager.rs` | NUM_EMBEDDERS constant | P1 |
| `crates/context-graph-utl/src/lifecycle/lambda.rs` | LifecycleLambdaWeights reference | P2 |

---

## 5. Scope

### 5.1 In Scope (What Remains)

1. **Add lambda weight clamping** to `MetaUtlTracker::update_weights()`
   - Clamp each weight to [0.1, 0.9]
   - Re-normalize after clamping to maintain sum=1.0

2. **Add consecutive low tracking** to `MetaUtlTracker`
   - Track consecutive cycles with accuracy < 0.7
   - Trigger escalation flag when count >= 10

3. **Add Domain enum** for domain-specific tracking
   - Code, Medical, Legal, Creative, Research, General

4. **Add MetaLearningEvent** for event logging
   - LambdaAdjustment, BayesianEscalation, AccuracyAlert

5. **Add SelfCorrectionConfig** with constitution defaults

### 5.2 Out of Scope

- MCP handler wiring (already implemented)
- Basic prediction/validation flow (already implemented)
- Full State Verification tests (already implemented)

---

## 6. Full State Verification (FSV) Requirements

### 6.1 Source of Truth

| Entity | Source of Truth | Location |
|--------|-----------------|----------|
| NUM_EMBEDDERS | `context_graph_core::johari::NUM_EMBEDDERS` | `crates/context-graph-core/src/johari/manager.rs` |
| Lambda bounds | Constitution YAML | `docs2/constitution.yaml` |
| Accuracy threshold | Constitution YAML (0.7) | `docs2/constitution.yaml` |
| Error threshold | Constitution YAML (0.2) | `docs2/constitution.yaml` |
| Escalation cycles | Constitution YAML (10) | `docs2/constitution.yaml` |

### 6.2 Execute & Inspect Requirements

After implementation, verify by running:

```bash
# 1. Type check entire workspace
cargo check --workspace

# 2. Run existing FSV tests
cargo test -p context-graph-mcp full_state_verification_meta_utl --no-fail-fast

# 3. Run clippy with strict settings
cargo clippy -p context-graph-mcp -- -D warnings

# 4. Verify no regressions
cargo test -p context-graph-mcp --lib
```

### 6.3 Boundary & Edge Case Audit

| Edge Case ID | Description | Before State | After State | Verification Method |
|--------------|-------------|--------------|-------------|---------------------|
| EC-001 | Weight below 0.1 after update | weights[0] = 0.05 | weights[0] = 0.1, re-normalized | Unit test with mock accuracy |
| EC-002 | Weight above 0.9 after update | weights[0] = 0.95 | weights[0] = 0.9, re-normalized | Unit test with mock accuracy |
| EC-003 | 10 consecutive low accuracy | consecutive_low = 9 | consecutive_low = 10, escalation=true | Integration test |
| EC-004 | Accuracy exactly at 0.7 threshold | accuracy = 0.7 | consecutive_low NOT incremented | Unit test boundary |
| EC-005 | All embedders at minimum (0.1) | 13 weights at 0.1 | Sum = 1.3, normalize to sum=1.0 | Unit test |
| EC-006 | Single embedder at 1.0, others at 0.0 | weights[0]=1.0, rest=0.0 | weights[0]=0.9, distribute 0.1 | Unit test |

### 6.4 Evidence of Success Logs

Implementation must produce logs at these checkpoints:

```rust
// Lambda clamping log
tracing::debug!(
    embedder_idx = %idx,
    original_weight = %before,
    clamped_weight = %after,
    "Lambda weight clamped to bounds"
);

// Escalation trigger log
tracing::warn!(
    consecutive_low = %self.consecutive_low_count,
    threshold = 10,
    "Bayesian escalation triggered"
);

// Weight update log
tracing::info!(
    validation_count = %self.validation_count,
    weights = ?self.current_weights,
    "Meta-UTL weights updated"
);
```

---

## 7. Manual Testing Requirements

### 7.1 Pre-Implementation Verification

```bash
# Confirm current state compiles
cargo build -p context-graph-mcp

# Confirm existing tests pass
cargo test -p context-graph-mcp meta_utl

# Get baseline FSV test count
cargo test -p context-graph-mcp full_state_verification_meta_utl --no-run 2>&1 | grep "test"
```

### 7.2 Synthetic Test Data

#### Test Data Set 1: Lambda Clamping Scenario

```rust
// Input: Create tracker with extreme accuracy distribution
let mut tracker = MetaUtlTracker::new();

// Embedder 0 gets 100% accuracy, others get 0%
for _ in 0..50 {
    tracker.record_accuracy(0, 1.0);  // Perfect
    for i in 1..13 {
        tracker.record_accuracy(i, 0.0);  // Terrible
    }
}
tracker.update_weights();

// Expected Output:
// weights[0] = 0.9 (clamped from ~1.0)
// weights[1..13] = 0.1/12 each (redistributed)
// sum(weights) = 1.0
assert!((tracker.current_weights.iter().sum::<f32>() - 1.0).abs() < 0.001);
assert!(tracker.current_weights[0] <= 0.9);
assert!(tracker.current_weights[0] >= 0.1);
```

#### Test Data Set 2: Escalation Trigger Scenario

```rust
// Input: 10 consecutive low accuracy cycles
let mut tracker = MetaUtlTracker::new();

for cycle in 0..10 {
    for embedder in 0..13 {
        tracker.record_accuracy(embedder, 0.5);  // Below 0.7
    }
}

// Expected Output:
// escalation_needed() returns true
// consecutive_low_count >= 10
assert!(tracker.needs_escalation());
assert_eq!(tracker.consecutive_low_count(), 10);
```

#### Test Data Set 3: Recovery Scenario

```rust
// Input: 9 low cycles, then 1 high cycle
let mut tracker = MetaUtlTracker::new();

for cycle in 0..9 {
    for embedder in 0..13 {
        tracker.record_accuracy(embedder, 0.5);
    }
}
// Now record high accuracy
for embedder in 0..13 {
    tracker.record_accuracy(embedder, 0.9);  // Above 0.7
}

// Expected Output:
// consecutive_low_count reset to 0
// escalation_needed() returns false
assert!(!tracker.needs_escalation());
assert_eq!(tracker.consecutive_low_count(), 0);
```

### 7.3 Database/State Verification

After running test suite, manually verify state:

```bash
# 1. Check no panics in test output
cargo test -p context-graph-mcp meta_utl 2>&1 | grep -i panic

# 2. Verify all FSV tests pass
cargo test -p context-graph-mcp full_state_verification_meta_utl -- --nocapture

# 3. Check for any warnings
cargo test -p context-graph-mcp 2>&1 | grep -i warning
```

---

## 8. Implementation Checklist

### 8.1 Phase 1: Modify MetaUtlTracker (core.rs)

- [ ] Add `consecutive_low_count: usize` field
- [ ] Add `escalation_triggered: bool` field
- [ ] Modify `record_accuracy()` to track consecutive low
- [ ] Modify `update_weights()` to clamp to [0.1, 0.9]
- [ ] Add `needs_escalation() -> bool` method
- [ ] Add `reset_consecutive_low()` method
- [ ] Add tracing logs for all state changes

### 8.2 Phase 2: Add Supporting Types

- [ ] Add `Domain` enum to `core.rs`
- [ ] Add `MetaLearningEventType` enum
- [ ] Add `MetaLearningEvent` struct
- [ ] Add `SelfCorrectionConfig` struct with Default

### 8.3 Phase 3: Tests

- [ ] Add unit test for lambda clamping (EC-001, EC-002)
- [ ] Add unit test for escalation trigger (EC-003)
- [ ] Add unit test for threshold boundary (EC-004)
- [ ] Add unit test for extreme distributions (EC-005, EC-006)
- [ ] Verify all existing FSV tests still pass

---

## 9. Verification Commands

```bash
# Full verification sequence
cargo check --workspace && \
cargo clippy -p context-graph-mcp -- -D warnings && \
cargo test -p context-graph-mcp meta_utl --no-fail-fast && \
cargo test -p context-graph-mcp full_state_verification_meta_utl --no-fail-fast && \
cargo doc -p context-graph-mcp --no-deps
```

---

## 10. Constraints

- **NO BACKWARDS COMPATIBILITY** - Fail fast with robust error logging
- **NO MOCK DATA** - Tests must use real MetaUtlTracker instances
- **NO unwrap()** - Use `expect()` with context or return `Result`
- **FAIL FAST** - Return errors immediately, do not silently ignore
- All accuracy values MUST be clamped to [0.0, 1.0]
- All weights MUST be clamped to [0.1, 0.9] (REQ-METAUTL-007)
- All weights MUST sum to 1.0 (REQ-METAUTL-006)
- All timestamps MUST use `std::time::Instant` for internal tracking

---

## 11. Rollback Plan

If implementation fails validation:

1. `git checkout -- crates/context-graph-mcp/src/handlers/core.rs`
2. Document failure reason in this task file under Notes section
3. Create follow-up task addressing specific issues
4. Do NOT attempt partial fixes - full rollback only

---

## 12. Notes

### 12.1 Architecture Decision

The original task proposed creating `crates/context-graph-utl/src/meta/types.rs`, but the actual implementation places Meta-UTL types in `crates/context-graph-mcp/src/handlers/core.rs`. This is intentional:

- MetaUtlTracker needs direct access to MCP request/response cycle
- Predictions are tied to MCP handlers, not standalone UTL processing
- Keeps all MCP state in one location for maintainability

### 12.2 Existing Test Coverage

FSV tests already exist covering:
- `test_fsv_learning_trajectory_all_embedders`
- `test_fsv_predict_storage_and_validate`
- Edge cases for invalid indices, unknown predictions

### 12.3 Git Recent History (for context)

```
b851ae6 feat(TASK-IDENTITY-P0-001): extend IdentityContinuity
664df8b feat(TASK-DREAM-P0-002): implement Poincare ball math
487e3eb feat(TASK-DREAM-P0-001): implement dream layer types
```

---

**Task History:**

| Version | Date | Author | Changes |
|---------|------|--------|---------|
| 1.0.0 | 2026-01-11 | ContextGraph Team | Initial task specification |
| 2.0.0 | 2026-01-11 | AI Agent | Updated with codebase audit, FSV requirements, manual testing, correct file paths |
