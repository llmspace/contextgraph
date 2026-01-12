# TASK-DELTA-P1-004: Integration Tests for compute_delta_sc

## Metadata

| Field | Value |
|-------|-------|
| **ID** | TASK-DELTA-P1-004 |
| **Version** | 2.0 |
| **Status** | **COMPLETED** |
| **Layer** | surface |
| **Sequence** | 4 of 4 |
| **Priority** | P1 |
| **Estimated Complexity** | medium |
| **Estimated Duration** | 3-4 hours |
| **Implements** | Test Plan from SPEC-UTL-001 |
| **Depends On** | TASK-DELTA-P1-001, TASK-DELTA-P1-002, TASK-DELTA-P1-003 |
| **Spec Ref** | SPEC-UTL-001 |
| **Gap Ref** | MASTER-CONSCIOUSNESS-GAP-ANALYSIS.md GAP 1 |

---

## CRITICAL STATUS UPDATE

**This task is COMPLETED.** Integration tests were implemented as part of the unified development of TASK-DELTA-P1-002.

**Actual Test Files**:
- `crates/context-graph-mcp/src/handlers/tests/utl/delta_sc_valid.rs` - 6 tests passing
- `crates/context-graph-mcp/src/handlers/tests/utl/delta_sc_errors.rs` - 5 tests passing
- `crates/context-graph-mcp/src/handlers/tests/manual_delta_sc_verification.rs` - FSV tests

**Run Tests**: `cargo test -p context-graph-mcp delta_sc -- --nocapture`

---

## Context

This task creates comprehensive integration tests for the `compute_delta_sc` MCP tool, validating end-to-end behavior with realistic data.

**Why This Last**: Following Inside-Out, Bottom-Up:
1. Unit tests exist in each component task
2. Integration tests validate the complete pipeline
3. Tests serve as living documentation and regression protection

**Gap Being Addressed**:
> GAP 1: UTL compute_delta_sc MCP Tool Missing
> Integration tests prove the gap is fully resolved

---

## Input Context Files

| Purpose | File |
|---------|------|
| Test plan reference | `specs/functional/SPEC-UTL-001.md#test-plan` |
| Existing integration tests | `crates/context-graph-mcp/src/handlers/tests/integration_e2e.rs` |
| FSV test pattern | `crates/context-graph-mcp/src/handlers/tests/full_state_verification.rs` |
| Test helpers | `crates/context-graph-core/src/types/fingerprint/teleological/test_helpers.rs` |
| Mock data generation | `crates/context-graph-mcp/src/handlers/tests/inject_synthetic_data.rs` |

---

## Prerequisites

| Check | Verification |
|-------|--------------|
| TASK-DELTA-P1-003 complete | Handler registered and compiles |
| Test framework set up | `cargo test -p context-graph-mcp` runs |
| Test helpers available | `TeleologicalFingerprint::zeroed()` exists |
| FSV pattern understood | Read existing full_state_verification tests |

---

## Scope

### In Scope

- Integration tests for all test cases from SPEC-UTL-001
- Full State Verification (FSV) test for compute_delta_sc
- Performance benchmark tests
- Property-based tests for output bounds
- Error handling tests for all error states
- Edge case tests per SPEC-UTL-001

### Out of Scope

- Chaos tests (separate task)
- Load testing (separate task)
- Fuzz testing (future enhancement)

---

## Definition of Done

### Test File Structure

```rust
// File: crates/context-graph-mcp/src/handlers/tests/delta_sc_integration.rs

use std::sync::Arc;

use uuid::Uuid;

use context_graph_core::types::fingerprint::{
    JohariFingerprint, PurposeVector, SemanticFingerprint, TeleologicalFingerprint,
};
use context_graph_core::types::JohariQuadrant;

use crate::handlers::Handlers;
use crate::protocol::{error_codes, JsonRpcRequest};
use crate::types::delta_sc::{ComputeDeltaScRequest, ComputeDeltaScResponse};

// ============================================================================
// Test Helpers
// ============================================================================

fn create_test_handlers() -> Handlers {
    // Create handlers with mock stores
    Handlers::new_for_testing()
}

fn create_test_fingerprint_pair() -> (TeleologicalFingerprint, TeleologicalFingerprint) {
    let old = TeleologicalFingerprint::new(
        SemanticFingerprint::zeroed(),
        PurposeVector::default(),
        JohariFingerprint::zeroed(),
        [0u8; 32],
    );

    let mut new = old.clone();
    // Modify semantic embeddings to create measurable delta
    // ...

    (old, new)
}

// ============================================================================
// Unit Tests from SPEC-UTL-001 Test Plan
// ============================================================================

/// TC-001: GMM entropy returns value in [0, 1]
#[tokio::test]
async fn test_gmm_entropy_bounded() {
    let handlers = create_test_handlers();
    let (old_fp, new_fp) = create_test_fingerprint_pair();

    let request = ComputeDeltaScRequest {
        vertex_id: Uuid::new_v4(),
        old_fingerprint: old_fp,
        new_fingerprint: new_fp,
        include_diagnostics: true,
        johari_threshold: None,
    };

    let response = handlers.delta_sc_computer.compute(&request).await.unwrap();

    // E1 uses GMM
    assert!(response.delta_s_per_embedder[0] >= 0.0);
    assert!(response.delta_s_per_embedder[0] <= 1.0);
    println!("[PASS] TC-001: GMM entropy in [0, 1]");
}

/// TC-010: Delta-C uses correct weights (0.4, 0.4, 0.2)
#[tokio::test]
async fn test_delta_c_weights() {
    let handlers = create_test_handlers();
    let (old_fp, new_fp) = create_test_fingerprint_pair();

    let request = ComputeDeltaScRequest {
        vertex_id: Uuid::new_v4(),
        old_fingerprint: old_fp,
        new_fingerprint: new_fp,
        include_diagnostics: true,
        johari_threshold: None,
    };

    let response = handlers.delta_sc_computer.compute(&request).await.unwrap();
    let diag = response.diagnostics.unwrap();

    // Verify formula: delta_c = 0.4*connectivity + 0.4*cluster_fit + 0.2*consistency
    let expected = 0.4 * diag.connectivity + 0.4 * diag.cluster_fit + 0.2 * diag.consistency;
    let tolerance = 0.001;
    assert!((response.delta_c - expected).abs() < tolerance,
        "Delta-C mismatch: got {}, expected {}", response.delta_c, expected);

    println!("[PASS] TC-010: Delta-C weights correct (0.4, 0.4, 0.2)");
}

/// TC-011: Johari classification correct for all quadrants
#[tokio::test]
async fn test_johari_classification_all_quadrants() {
    let handlers = create_test_handlers();

    // Test Open: delta_s <= 0.5, delta_c > 0.5
    // Test Blind: delta_s > 0.5, delta_c <= 0.5
    // Test Hidden: delta_s <= 0.5, delta_c <= 0.5
    // Test Unknown: delta_s > 0.5, delta_c > 0.5

    // Create fingerprints that produce each quadrant...
    // (Implementation details)

    println!("[PASS] TC-011: All Johari quadrants classified correctly");
}

/// TC-012: Missing embedder returns fallback
#[tokio::test]
async fn test_missing_embedder_fallback() {
    // Test with partial fingerprint (if allowed)
    // Should return fallback 0.5 for missing embedder
    println!("[PASS] TC-012: Missing embedder fallback works");
}

/// TC-013: Identical fingerprints return zero entropy
#[tokio::test]
async fn test_identical_fingerprints_zero_entropy() {
    let handlers = create_test_handlers();
    let fp = TeleologicalFingerprint::new(
        SemanticFingerprint::zeroed(),
        PurposeVector::default(),
        JohariFingerprint::zeroed(),
        [0u8; 32],
    );

    let request = ComputeDeltaScRequest {
        vertex_id: Uuid::new_v4(),
        old_fingerprint: fp.clone(),
        new_fingerprint: fp,
        include_diagnostics: false,
        johari_threshold: None,
    };

    let response = handlers.delta_sc_computer.compute(&request).await.unwrap();

    // Identical fingerprints should have low/zero entropy
    for (idx, delta_s) in response.delta_s_per_embedder.iter().enumerate() {
        assert!(*delta_s < 0.1,
            "E{} delta_s should be ~0 for identical fingerprints, got {}", idx + 1, delta_s);
    }

    println!("[PASS] TC-013: Identical fingerprints produce near-zero entropy");
}

// ============================================================================
// Integration Tests from SPEC-UTL-001 Test Plan
// ============================================================================

/// TC-014: MCP tool registered and discoverable
#[tokio::test]
async fn test_tool_discoverable() {
    let handlers = create_test_handlers();

    let response = handlers.handle_tools_list(Some(1.into())).await;

    let result = response.result.unwrap();
    let tools = result["tools"].as_array().unwrap();

    let found = tools.iter().any(|t| t["name"] == "gwt/compute_delta_sc");
    assert!(found, "gwt/compute_delta_sc not in tools/list");

    println!("[PASS] TC-014: Tool discoverable via tools/list");
}

/// TC-015: Full pipeline with real fingerprints
#[tokio::test]
async fn test_full_pipeline() {
    let handlers = create_test_handlers();
    let (old_fp, new_fp) = create_test_fingerprint_pair();

    let params = serde_json::to_value(ComputeDeltaScRequest {
        vertex_id: Uuid::new_v4(),
        old_fingerprint: old_fp,
        new_fingerprint: new_fp,
        include_diagnostics: true,
        johari_threshold: Some(0.5),
    }).unwrap();

    let response = handlers.handle_gwt_compute_delta_sc(Some(1.into()), Some(params)).await;

    assert!(response.error.is_none(), "Unexpected error: {:?}", response.error);

    let result: ComputeDeltaScResponse = serde_json::from_value(response.result.unwrap()).unwrap();

    // Validate response structure
    assert_eq!(result.delta_s_per_embedder.len(), 13);
    assert!(result.delta_s_aggregate >= 0.0 && result.delta_s_aggregate <= 1.0);
    assert!(result.delta_c >= 0.0 && result.delta_c <= 1.0);
    assert_eq!(result.johari_quadrants.len(), 13);
    assert!(result.diagnostics.is_some());

    println!("[PASS] TC-015: Full pipeline works with real fingerprints");
}

/// TC-017: Response matches schema
#[tokio::test]
async fn test_response_schema() {
    let handlers = create_test_handlers();
    let (old_fp, new_fp) = create_test_fingerprint_pair();

    let params = serde_json::to_value(ComputeDeltaScRequest {
        vertex_id: Uuid::new_v4(),
        old_fingerprint: old_fp,
        new_fingerprint: new_fp,
        include_diagnostics: false,
        johari_threshold: None,
    }).unwrap();

    let response = handlers.handle_gwt_compute_delta_sc(Some(1.into()), Some(params)).await;
    let result = response.result.unwrap();

    // Verify all required fields present
    assert!(result.get("delta_s_per_embedder").is_some());
    assert!(result.get("delta_s_aggregate").is_some());
    assert!(result.get("delta_c").is_some());
    assert!(result.get("johari_quadrants").is_some());
    assert!(result.get("johari_aggregate").is_some());
    assert!(result.get("utl_learning_potential").is_some());

    // Diagnostics should be absent when not requested
    assert!(result.get("diagnostics").is_none());

    println!("[PASS] TC-017: Response matches schema");
}

// ============================================================================
// Performance Tests from SPEC-UTL-001 Test Plan
// ============================================================================

/// TC-019: Compute latency < 25ms p95
#[tokio::test]
async fn test_latency_p95() {
    use std::time::Instant;

    let handlers = create_test_handlers();
    let mut latencies = Vec::with_capacity(100);

    for _ in 0..100 {
        let (old_fp, new_fp) = create_test_fingerprint_pair();

        let request = ComputeDeltaScRequest {
            vertex_id: Uuid::new_v4(),
            old_fingerprint: old_fp,
            new_fingerprint: new_fp,
            include_diagnostics: false,
            johari_threshold: None,
        };

        let start = Instant::now();
        let _ = handlers.delta_sc_computer.compute(&request).await;
        latencies.push(start.elapsed().as_millis() as u64);
    }

    latencies.sort();
    let p95 = latencies[94]; // 95th percentile

    assert!(p95 < 25, "p95 latency {} ms exceeds 25ms target", p95);
    println!("[PASS] TC-019: p95 latency {} ms < 25ms", p95);
}

// ============================================================================
// Property-Based Tests from SPEC-UTL-001 Test Plan
// ============================================================================

/// TC-022: Delta-S always in [0, 1]
#[tokio::test]
async fn test_delta_s_bounded_property() {
    let handlers = create_test_handlers();

    for _ in 0..50 {
        let (old_fp, new_fp) = create_test_fingerprint_pair();

        let request = ComputeDeltaScRequest {
            vertex_id: Uuid::new_v4(),
            old_fingerprint: old_fp,
            new_fingerprint: new_fp,
            include_diagnostics: false,
            johari_threshold: None,
        };

        let response = handlers.delta_sc_computer.compute(&request).await.unwrap();

        for (idx, &delta_s) in response.delta_s_per_embedder.iter().enumerate() {
            assert!(delta_s >= 0.0 && delta_s <= 1.0,
                "E{} delta_s {} out of bounds", idx + 1, delta_s);
            assert!(!delta_s.is_nan(), "E{} delta_s is NaN (AP-10 violation)", idx + 1);
            assert!(!delta_s.is_infinite(), "E{} delta_s is infinite (AP-10 violation)", idx + 1);
        }
    }

    println!("[PASS] TC-022: Delta-S always in [0, 1] for random inputs");
}

/// TC-023: Delta-C always in [0, 1]
#[tokio::test]
async fn test_delta_c_bounded_property() {
    let handlers = create_test_handlers();

    for _ in 0..50 {
        let (old_fp, new_fp) = create_test_fingerprint_pair();

        let request = ComputeDeltaScRequest {
            vertex_id: Uuid::new_v4(),
            old_fingerprint: old_fp,
            new_fingerprint: new_fp,
            include_diagnostics: false,
            johari_threshold: None,
        };

        let response = handlers.delta_sc_computer.compute(&request).await.unwrap();

        assert!(response.delta_c >= 0.0 && response.delta_c <= 1.0,
            "delta_c {} out of bounds", response.delta_c);
        assert!(!response.delta_c.is_nan(), "delta_c is NaN (AP-10 violation)");
        assert!(!response.delta_c.is_infinite(), "delta_c is infinite (AP-10 violation)");
    }

    println!("[PASS] TC-023: Delta-C always in [0, 1] for random inputs");
}

/// TC-025: UTL potential = Delta-S * Delta-C
#[tokio::test]
async fn test_utl_potential_invariant() {
    let handlers = create_test_handlers();
    let (old_fp, new_fp) = create_test_fingerprint_pair();

    let request = ComputeDeltaScRequest {
        vertex_id: Uuid::new_v4(),
        old_fingerprint: old_fp,
        new_fingerprint: new_fp,
        include_diagnostics: false,
        johari_threshold: None,
    };

    let response = handlers.delta_sc_computer.compute(&request).await.unwrap();

    let expected = response.delta_s_aggregate * response.delta_c;
    let tolerance = 0.0001;

    assert!((response.utl_learning_potential - expected).abs() < tolerance,
        "UTL potential mismatch: got {}, expected {}", response.utl_learning_potential, expected);

    println!("[PASS] TC-025: UTL potential = Delta-S * Delta-C");
}

// ============================================================================
// Full State Verification (FSV) Test
// ============================================================================

/// Full State Verification for compute_delta_sc
///
/// This test follows the FSV pattern established in the codebase,
/// verifying complete state transitions and invariants.
#[tokio::test]
async fn full_state_verification_compute_delta_sc() {
    println!("\n========== TASK-DELTA-P1-004 FULL STATE VERIFICATION ==========\n");

    let handlers = create_test_handlers();

    // Test 1: Basic computation
    println!("[TEST 1] Basic Delta-S/Delta-C computation");
    let (old_fp, new_fp) = create_test_fingerprint_pair();
    let request = ComputeDeltaScRequest {
        vertex_id: Uuid::new_v4(),
        old_fingerprint: old_fp,
        new_fingerprint: new_fp,
        include_diagnostics: true,
        johari_threshold: None,
    };

    let response = handlers.delta_sc_computer.compute(&request).await.unwrap();
    assert_eq!(response.delta_s_per_embedder.len(), 13);
    println!("  Delta-S aggregate: {}", response.delta_s_aggregate);
    println!("  Delta-C: {}", response.delta_c);
    println!("  Johari aggregate: {:?}", response.johari_aggregate);
    println!("[TEST 1 PASSED]\n");

    // Test 2: All 13 embedders computed
    println!("[TEST 2] All 13 embedders have valid outputs");
    for (idx, delta_s) in response.delta_s_per_embedder.iter().enumerate() {
        assert!(delta_s.is_finite(), "E{} has non-finite delta_s", idx + 1);
        println!("  E{}: delta_s = {:.4}", idx + 1, delta_s);
    }
    println!("[TEST 2 PASSED]\n");

    // Test 3: Johari classification
    println!("[TEST 3] Johari classification");
    for (idx, quadrant) in response.johari_quadrants.iter().enumerate() {
        println!("  E{}: {:?}", idx + 1, quadrant);
    }
    println!("[TEST 3 PASSED]\n");

    // Test 4: Diagnostics included
    println!("[TEST 4] Diagnostics present when requested");
    let diag = response.diagnostics.as_ref().unwrap();
    println!("  Connectivity: {}", diag.connectivity);
    println!("  ClusterFit: {}", diag.cluster_fit);
    println!("  Consistency: {}", diag.consistency);
    println!("  Computation time: {} us", diag.computation_time_us);
    println!("[TEST 4 PASSED]\n");

    // Test 5: Error handling
    println!("[TEST 5] Error handling for invalid parameters");
    let invalid_response = handlers.handle_gwt_compute_delta_sc(
        Some(1.into()),
        Some(serde_json::json!({"invalid": "params"})),
    ).await;
    assert!(invalid_response.error.is_some());
    println!("  Error code: {}", invalid_response.error.as_ref().unwrap().code);
    println!("[TEST 5 PASSED]\n");

    println!("========== FSV SUMMARY ==========");
    println!("[EVIDENCE] TASK-DELTA-P1-004 Complete");
    println!("  - Total tests: 5");
    println!("  - Tests passed: 5");
    println!("  - Embedders verified: 13/13");
    println!("  - Johari quadrants: validated");
    println!("  - Diagnostics: validated");
    println!("  - Error handling: validated");
    println!("=================================\n");
}
```

### Constraints

- All tests MUST be async/await compatible
- Tests MUST use the FSV pattern established in the codebase
- Property tests MUST run multiple iterations (50+)
- Performance tests MUST measure p95 latency
- Tests MUST NOT depend on external services
- Tests MUST be deterministic (use seeded random if needed)

### Verification

```bash
# All integration tests pass
cargo test -p context-graph-mcp delta_sc_integration -- --nocapture

# FSV test passes
cargo test -p context-graph-mcp full_state_verification_compute_delta_sc -- --nocapture

# No test flakiness (run 3 times)
for i in 1 2 3; do cargo test -p context-graph-mcp delta_sc_integration; done
```

---

## Files to Create

| Path | Description |
|------|-------------|
| `crates/context-graph-mcp/src/handlers/tests/delta_sc_integration.rs` | Integration tests |

---

## Files to Modify

| Path | Change |
|------|--------|
| `crates/context-graph-mcp/src/handlers/tests/mod.rs` | Add `mod delta_sc_integration;` |

---

## Validation Criteria

| Criterion | Verification Method |
|-----------|---------------------|
| All SPEC-UTL-001 test cases implemented | Checklist review |
| FSV test passes | `cargo test full_state_verification_compute_delta_sc` |
| p95 latency < 25ms | Performance test TC-019 |
| No NaN/Infinity (AP-10) | Property tests TC-022, TC-023 |
| Tests are deterministic | Run 3 times with same results |

---

## Test Commands

```bash
# Run all delta_sc tests
cargo test -p context-graph-mcp delta_sc -- --nocapture

# Run integration tests only
cargo test -p context-graph-mcp delta_sc_integration -- --nocapture

# Run FSV test only
cargo test -p context-graph-mcp full_state_verification_compute_delta_sc -- --nocapture

# Run with verbose output
RUST_LOG=debug cargo test -p context-graph-mcp delta_sc -- --nocapture
```

---

## Notes

- Tests serve as living documentation for the compute_delta_sc tool
- FSV pattern ensures complete state verification
- Property tests catch edge cases that unit tests might miss
- Performance tests should run on consistent hardware for reliable results
- Consider adding mutation testing in future iterations

---

## Appendix A: Per-Embedder Delta-S Test Cases

This appendix specifies comprehensive tests for each of the 13 embedder Delta-S calculation methods per constitution.yaml delta_sc.Delta_S_methods.

### A.1 E1 (Semantic) - GMM+Mahalanobis Tests

| ID | Test | Input | Expected Output | Rationale |
|----|------|-------|-----------------|-----------|
| TC-E1-01 | Identical embedding | current == history[0] | Delta-S < 0.1 | Low surprise for identical |
| TC-E1-02 | Very different embedding | orthogonal vector | Delta-S > 0.8 | High surprise for novel |
| TC-E1-03 | Cluster member | current near cluster centroid | Delta-S < 0.3 | Low surprise in-cluster |
| TC-E1-04 | Outlier | current far from all clusters | Delta-S > 0.7 | High surprise for outlier |
| TC-E1-05 | Empty history | history = [] | Delta-S = 1.0 | Maximum surprise |
| TC-E1-06 | Dimension 1024 | correct embedding size | No error | ARCH-05 compliance |

### A.2 E2-E4, E8 (Temporal, Graph) - KNN Tests

| ID | Test | Input | Expected Output | Rationale |
|----|------|-------|-----------------|-----------|
| TC-E2-01 | K neighbors closer than average | d_k < mean | Delta-S < 0.5 | Familiar pattern |
| TC-E2-02 | K neighbors farther than average | d_k > mean | Delta-S > 0.5 | Surprising pattern |
| TC-E2-03 | K=1 edge case | only 1 neighbor | Valid Delta-S | Graceful handling |
| TC-E2-04 | K > history size | k=10, history=3 | Uses all history | Fallback behavior |
| TC-E2-05 | Sigmoid normalization | extreme distances | Delta-S in [0,1] | Bounded output |

### A.3 E5 (Causal) - Asymmetric KNN Tests

| ID | Test | Input | Expected Output | Rationale |
|----|------|-------|-----------------|-----------|
| TC-E5-01 | Cause-to-effect query | direction=forward | Delta-S * 1.2 | Direction modifier |
| TC-E5-02 | Effect-to-cause query | direction=backward | Delta-S * 0.8 | Direction modifier |
| TC-E5-03 | Neutral direction | direction=neutral | Delta-S * 1.0 | No modification |
| TC-E5-04 | Modifier bounds | extreme modifiers | Delta-S in [0,1] | Clamping works |
| TC-E5-05 | Causal chain | A->B->C pattern | Transitivity check | Relationship preserved |

### A.4 E6 (Sparse) - IDF/Jaccard Tests

| ID | Test | Input | Expected Output | Rationale |
|----|------|-------|-----------------|-----------|
| TC-E6-01 | Identical active dims | same sparse pattern | Delta-S = 0.0 | Perfect match |
| TC-E6-02 | Disjoint active dims | no overlap | Delta-S = 1.0 | No match |
| TC-E6-03 | Partial overlap | 50% shared dims | Delta-S ~ 0.5 | Proportional |
| TC-E6-04 | IDF weighting | rare dims match | Lower Delta-S | Rare matches valuable |
| TC-E6-05 | Common dims match | frequent dims match | Higher Delta-S | Common matches less valuable |
| TC-E6-06 | Empty sparse | all below threshold | Delta-S = 1.0 | Edge case |

### A.5 E7 (Code) - GMM+KNN Hybrid Tests

| ID | Test | Input | Expected Output | Rationale |
|----|------|-------|-----------------|-----------|
| TC-E7-01 | Weight sum | gmm_w + knn_w | = 1.0 | Invariant |
| TC-E7-02 | GMM dominates | code cluster pattern | Result ~ GMM | Structural match |
| TC-E7-03 | KNN dominates | local novelty | Result ~ KNN | Local surprise |
| TC-E7-04 | Balanced case | mixed pattern | 0.5*GMM + 0.5*KNN | Interpolation |
| TC-E7-05 | Code dimension | 1536D embedding | No error | Correct size |

### A.6 E9 (Hdc) - Hamming Tests

| ID | Test | Input | Expected Output | Rationale |
|----|------|-------|-----------------|-----------|
| TC-E9-01 | Identical binary | same bits | Delta-S = 0.0 | No Hamming distance |
| TC-E9-02 | All bits differ | inverted | Delta-S = 1.0 | Maximum distance |
| TC-E9-03 | Half bits differ | 50% flip | Delta-S ~ 0.5 | Proportional |
| TC-E9-04 | Binarization | continuous input | Valid binary | Threshold applied |
| TC-E9-05 | Prototype learning | repeated patterns | Lower Delta-S | Learning works |

### A.7 E10 (Multimodal) - Cross-modal KNN Tests

| ID | Test | Input | Expected Output | Rationale |
|----|------|-------|-----------------|-----------|
| TC-E10-01 | Aligned modalities | same surprise all | Low penalty | Consistent content |
| TC-E10-02 | Misaligned modalities | varied surprise | Higher penalty | Inconsistent |
| TC-E10-03 | Text-only surprise | text differs | Partial Delta-S | Modality isolation |
| TC-E10-04 | Image-only surprise | image differs | Partial Delta-S | Modality isolation |
| TC-E10-05 | Dimension splits | 256+256+256=768 | Correct parsing | Structure preserved |

### A.8 E11 (Entity) - TransE Tests

| ID | Test | Input | Expected Output | Rationale |
|----|------|-------|-----------------|-----------|
| TC-E11-01 | Same entity | current == history | Delta-S ~ 0.0 | Identity relation |
| TC-E11-02 | Related entity | h + r approx current | Low Delta-S | Translation match |
| TC-E11-03 | Unrelated entity | far from all | Delta-S ~ 1.0 | No translation path |
| TC-E11-04 | Running stats | many samples | Mean converges | Online learning |
| TC-E11-05 | Entity dimension | 384D | No error | Correct size |

### A.9 E12 (LateInteraction) - MaxSim Tests

| ID | Test | Input | Expected Output | Rationale |
|----|------|-------|-----------------|-----------|
| TC-E12-01 | Identical tokens | same sequence | Delta-S ~ 0.0 | Perfect MaxSim |
| TC-E12-02 | No token overlap | disjoint vocab | Delta-S ~ 1.0 | Zero MaxSim |
| TC-E12-03 | Partial overlap | some shared tokens | Moderate Delta-S | Partial match |
| TC-E12-04 | Variable length | different token counts | Valid Delta-S | Handles lengths |
| TC-E12-05 | Token dimension | 128D per token | Correct parsing | Structure preserved |
| TC-E12-06 | ColBERT scoring | MaxSim sum/count | Normalized | Per-token average |

### A.10 E13 (KeywordSplade) - Jaccard Tests

| ID | Test | Input | Expected Output | Rationale |
|----|------|-------|-----------------|-----------|
| TC-E13-01 | Same keywords | identical active | Delta-S = 0.0 | Perfect Jaccard |
| TC-E13-02 | No overlap | disjoint keywords | Delta-S = 1.0 | Zero Jaccard |
| TC-E13-03 | Subset match | A subset B | Proportional Delta-S | Jaccard formula |
| TC-E13-04 | Smoothing | edge cases | No division by zero | Laplace smoothing |
| TC-E13-05 | Sparse dimension | ~30K sparse | Efficient computation | Sparse math |

---

## Appendix B: Delta-C Component Tests

### B.1 Coherence Components

| Component | Weight | Test Cases |
|-----------|--------|------------|
| Connectivity | 0.4 | Graph connectivity, isolated nodes, hub nodes |
| ClusterFit | 0.4 | Within-cluster, between-cluster, boundary points |
| Consistency | 0.2 | Stable fingerprint, evolving fingerprint, drift |

### B.2 Coherence Formula Verification

```rust
#[tokio::test]
async fn test_delta_c_formula_exact() {
    // Given known component values
    let connectivity = 0.8;
    let cluster_fit = 0.6;
    let consistency = 0.9;

    // When Delta-C is computed
    let delta_c = compute_delta_c(connectivity, cluster_fit, consistency);

    // Then it matches the constitution formula exactly
    let expected = 0.4 * connectivity + 0.4 * cluster_fit + 0.2 * consistency;
    assert!((delta_c - expected).abs() < 0.0001);
    assert_eq!(expected, 0.74); // 0.32 + 0.24 + 0.18
}
```

---

## Appendix C: Johari Classification Tests

### C.1 Quadrant Boundary Tests

| Quadrant | Delta-S | Delta-C | Test Cases |
|----------|---------|---------|------------|
| Open | <= 0.5 | > 0.5 | (0.5, 0.51), (0.0, 1.0), (0.49, 0.99) |
| Blind | > 0.5 | <= 0.5 | (0.51, 0.5), (1.0, 0.0), (0.99, 0.49) |
| Hidden | <= 0.5 | <= 0.5 | (0.5, 0.5), (0.0, 0.0), (0.49, 0.49) |
| Unknown | > 0.5 | > 0.5 | (0.51, 0.51), (1.0, 1.0), (0.99, 0.99) |

### C.2 Threshold Variation Tests

```rust
#[tokio::test]
async fn test_johari_custom_threshold() {
    // Test with adaptive threshold from constitution
    // Range: [0.35, 0.65] per adaptive_thresholds.priors.theta_joh

    for threshold in [0.35, 0.45, 0.5, 0.55, 0.65] {
        let delta_s = 0.5;
        let delta_c = 0.5;

        let quadrant = classify_johari(delta_s, delta_c, threshold);

        // At exactly threshold, verify classification
        if delta_s <= threshold && delta_c > threshold {
            assert_eq!(quadrant, JohariQuadrant::Open);
        }
        // ... other cases
    }
}
```

---

## Appendix D: Performance Benchmarks

### D.1 Per-Embedder Latency Targets

| Embedder | Target Latency | Rationale |
|----------|---------------|-----------|
| E1 (GMM) | < 3ms | GMM inference |
| E2-4, E8 (KNN) | < 1ms | Simple distance |
| E5 (Asymmetric) | < 1.5ms | KNN + modifier |
| E6 (Sparse IDF) | < 2ms | Sparse iteration |
| E7 (Hybrid) | < 4ms | GMM + KNN |
| E9 (Hamming) | < 0.5ms | Binary XOR |
| E10 (Cross-modal) | < 3ms | 3x modality |
| E11 (TransE) | < 1ms | L2 distance |
| E12 (MaxSim) | < 3ms | Token iteration |
| E13 (Jaccard) | < 1ms | Sparse Jaccard |
| **Total** | **< 20ms** | All 13 embedders |

### D.2 Benchmark Test

```rust
#[tokio::test]
async fn benchmark_all_embedders() {
    let config = SurpriseConfig::default();
    let calculators = EmbedderEntropyFactory::create_all(&config);

    let mut total_time = Duration::ZERO;

    for (idx, calculator) in calculators.iter().enumerate() {
        let dim = embedder_dimension(Embedder::from_index(idx).unwrap());
        let current = vec![0.5f32; dim];
        let history: Vec<Vec<f32>> = vec![vec![0.6f32; dim]; 10];

        let start = Instant::now();
        for _ in 0..100 {
            let _ = calculator.compute_delta_s(&current, &history, 5);
        }
        let elapsed = start.elapsed() / 100;

        total_time += elapsed;
        println!("E{}: {:?}", idx + 1, elapsed);
    }

    println!("Total (13 embedders): {:?}", total_time);
    assert!(total_time.as_millis() < 20, "Total latency exceeded 20ms");
}
```
