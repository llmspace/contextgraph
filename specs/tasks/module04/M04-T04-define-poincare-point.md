---
id: "M04-T04"
title: "Define PoincarePoint for 64D Hyperbolic Space"
version: "2.1.0"
updated: "2026-01-03"
layer: "foundation"
status: "complete"
priority: "critical"
estimated_hours: 2
sequence: 7
depends_on:
  - "M04-T02"  # HyperbolicConfig - VERIFIED COMPLETE
spec_refs:
  - "TECH-GRAPH-004 Section 5.1"
  - "REQ-KG-050"
  - "REQ-KG-054"
files_created:
  - path: "crates/context-graph-graph/src/hyperbolic/poincare.rs"
    description: "PoincarePoint struct implementation"
files_modified:
  - path: "crates/context-graph-graph/src/hyperbolic/mod.rs"
    description: "Added `pub mod poincare;` and re-export PoincarePoint"
test_file: "crates/context-graph-graph/src/hyperbolic/poincare.rs"
completion:
  verified_by: "sherlock-holmes"
  commit: "af8a55e"
  date: "2026-01-03"
  tests_passed: 30
---

# M04-T04: Define PoincarePoint for 64D Hyperbolic Space

## STATUS: ✅ COMPLETE

**Completed**: 2026-01-03
**Commit**: `af8a55e feat(graph): complete M04-T03 ConeConfig and M04-T04 PoincarePoint`
**Tests**: 30/30 PASS

## Implementation Summary

### Files Created/Modified
| File | Status | Description |
|------|--------|-------------|
| `crates/context-graph-graph/src/hyperbolic/poincare.rs` | ✅ Created | 639 lines, PoincarePoint struct with 30 unit tests |
| `crates/context-graph-graph/src/hyperbolic/mod.rs` | ✅ Updated | Module declaration + re-export |
| `crates/context-graph-graph/src/lib.rs` | ✅ Updated | Root re-export at line 51 |

### Struct Signature
```rust
#[repr(C, align(64))]
#[derive(Clone, Debug)]
pub struct PoincarePoint {
    pub coords: [f32; 64],
}
```

### Methods Implemented
| Method | Signature | Purpose |
|--------|-----------|---------|
| `origin()` | `fn origin() -> Self` | Creates origin (all zeros) |
| `from_coords()` | `fn from_coords(coords: [f32; 64]) -> Self` | Unchecked construction |
| `from_coords_projected()` | `fn from_coords_projected(coords: [f32; 64], config: &HyperbolicConfig) -> Self` | Validated construction |
| `norm_squared()` | `fn norm_squared(&self) -> f32` | Σ(coords[i]²) |
| `norm()` | `fn norm(&self) -> f32` | sqrt(norm_squared) |
| `project()` | `fn project(&mut self, config: &HyperbolicConfig)` | Rescale to max_norm |
| `projected()` | `fn projected(&self, config: &HyperbolicConfig) -> Self` | Non-mutating project |
| `is_valid()` | `fn is_valid(&self) -> bool` | norm_squared < 1.0 |
| `is_valid_for_config()` | `fn is_valid_for_config(&self, config: &HyperbolicConfig) -> bool` | norm < max_norm |

### Memory Layout Verified
- **Size**: 256 bytes (64 × f32)
- **Alignment**: 64 bytes (cache line aligned for SIMD)
- **repr(C)**: FFI-compatible for CUDA kernels (M04-T23)

---

## Verification Evidence

### Build Verification
```bash
$ cargo build -p context-graph-graph
# Compiles successfully (exit 0)
```

### Test Verification
```bash
$ cargo test -p context-graph-graph poincare
# 30 tests passed, 0 failed
```

### Tests Passing
- test_origin_is_zero_vector
- test_origin_has_zero_norm
- test_default_is_origin
- test_from_coords_preserves_values
- test_from_coords_projected_ensures_validity
- test_norm_squared_single_nonzero
- test_norm_squared_multiple_nonzero
- test_norm_pythagorean
- test_norm_uniform_coords
- test_project_inside_ball_unchanged
- test_project_outside_ball_rescaled
- test_project_at_boundary
- test_projected_returns_new_point
- test_project_preserves_direction
- test_is_valid_origin
- test_is_valid_inside_ball
- test_is_valid_at_boundary_false
- test_is_valid_outside_ball_false
- test_is_valid_for_config
- test_size_is_256_bytes
- test_alignment_is_64_bytes
- test_equality_same_coords
- test_inequality_different_coords
- test_clone_independent
- test_edge_case_very_small_norm
- test_edge_case_near_max_norm
- test_edge_case_negative_coords
- test_edge_case_project_zero_vector
- test_edge_case_nan_detection
- test_edge_case_infinity

---

## Acceptance Criteria ✅

- [x] `crates/context-graph-graph/src/hyperbolic/poincare.rs` exists
- [x] `PoincarePoint` struct has `coords: [f32; 64]`
- [x] `#[repr(C, align(64))]` attribute present
- [x] `origin()` returns all zeros
- [x] `from_coords()` takes `[f32; 64]`
- [x] `norm_squared()` computes sum of squares
- [x] `norm()` computes sqrt of norm_squared
- [x] `project(&HyperbolicConfig)` rescales when norm >= max_norm
- [x] `projected(&HyperbolicConfig)` returns new projected point
- [x] `is_valid()` returns true when norm_squared < 1.0
- [x] `Clone`, `Debug`, `PartialEq` traits implemented
- [x] `Default` implements origin()
- [x] Size is exactly 256 bytes
- [x] Alignment is exactly 64 bytes
- [x] `cargo build -p context-graph-graph` succeeds
- [x] `cargo test -p context-graph-graph poincare` all pass
- [x] `cargo clippy -p context-graph-graph -- -D warnings` no warnings in poincare.rs

---

## What This Task Enables

Now unblocked:
- **M04-T05**: PoincareBall Mobius operations (mobius_add, distance, exp_map, log_map)
- **M04-T06**: EntailmentCone struct (uses PoincarePoint for apex)
- **M04-T23**: CUDA Poincare distance kernel (batch operations on PoincarePoint)

---

## Dependencies Satisfied

| Task | Status | Location |
|------|--------|----------|
| M04-T00 | ✅ Complete | Crate exists at `crates/context-graph-graph/` |
| M04-T02 | ✅ Complete | `HyperbolicConfig` at `src/config.rs:114-330` |
| M04-T02a | ✅ Complete | `validate()` method at `src/config.rs:246-306` |

---

## Constitution Compliance

| Requirement | Implementation |
|-------------|----------------|
| hyperbolic.dim = 64 | `coords: [f32; 64]` |
| hyperbolic.max_norm = 0.99999 | Used via `HyperbolicConfig::default()` |
| perf.latency.entailment_check < 1ms | Achieved via O(64) operations |
| No unwrap() in prod | ✅ No unwrap calls |
| repr(C) for FFI | `#[repr(C, align(64))]` |

---

## NO BACKWARDS COMPATIBILITY APPLIED

- No fallback implementations
- No mock data in tests (all tests use real PoincarePoint instances)
- No compatibility shims
- Tests fail fast on invalid state (NaN, Infinity correctly detected as invalid)
