# TASK-EMB-003: Create ProjectionError Enum

<task_spec id="TASK-EMB-003" version="2.0" updated="2026-01-06">

## Metadata

| Field | Value |
|-------|-------|
| **Task ID** | TASK-EMB-003 |
| **Title** | Create ProjectionError Enum |
| **Status** | **COMPLETE** (2026-01-06) |
| **Layer** | foundation |
| **Sequence** | 3 |
| **Implements** | REQ-EMB-001 |
| **Depends On** | TASK-EMB-002 (VERIFIED COMPLETE 2026-01-06) |
| **Estimated Complexity** | low |
| **Constitution Ref** | `AP-007` (fail fast, no stubs), Error Taxonomy from SPEC-EMB-001 |

---

## Current Codebase State (Verified 2026-01-06)

### Prerequisite TASK-EMB-002: COMPLETE

The `ProjectionMatrix` struct exists and is verified:

```
File: crates/context-graph-embeddings/src/models/pretrained/sparse/projection.rs
Lines: 171
Size: 5479 bytes
```

**Verified Contents:**
- `pub struct ProjectionMatrix` with 3 fields: `weights: Tensor`, `device: Device`, `weight_checksum: [u8; 32]`
- Constants: `EXPECTED_SHAPE = (30522, 1536)`, `EXPECTED_FILE_SIZE = 187_527_168`
- Module constants: `PROJECTION_WEIGHT_FILE`, `PROJECTION_TENSOR_NAME`
- 6 accessor methods implemented
- 4 unit tests passing
- Exported in `sparse/mod.rs` line 61

### What Does NOT Exist Yet (Your Task)

`ProjectionError` enum does NOT exist in the codebase. This is what you must create.

**Verification:**
```bash
grep -rn "ProjectionError" crates/context-graph-embeddings/src/
# Returns: empty (enum does not exist)
```

### Dependency: thiserror Crate

Available as workspace dependency:
```toml
# crates/context-graph-embeddings/Cargo.toml
thiserror = { workspace = true }
```

### Existing Error Patterns in Crate

Reference for consistent style (see `crates/context-graph-embeddings/src/gpu/model_loader/error.rs`):
```rust
use thiserror::Error;

#[derive(Debug, Error)]
pub enum ModelLoadError {
    #[error("GPU initialization failed: {message}")]
    GpuInitError { message: String },
    // ... etc
}
```

---

## Problem Statement

The projection module needs specific error types for clear error messages when:
1. Weight file is missing
2. Weight file checksum doesn't match
3. Weight matrix has wrong dimensions
4. GPU operation fails
5. Projection is used before initialization

**Constitution Requirement (AP-007):**
> "Stub data in prod → use tests/fixtures/"

This means: If weight file is missing, **FAIL FAST with clear error**. Do NOT fall back to hash-based projection.

---

## Exact File to Modify

```
crates/context-graph-embeddings/src/models/pretrained/sparse/projection.rs
```

**Current state:** File exists with `ProjectionMatrix` struct (171 lines).
**Your task:** Add `ProjectionError` enum AFTER the existing code.

---

## Exact Code to Add

Add this code at the END of `projection.rs`, BEFORE the `#[cfg(test)]` module:

```rust
use std::path::PathBuf;
use thiserror::Error;

/// Errors that can occur during sparse projection.
///
/// # Error Codes (per SPEC-EMB-001)
/// - EMB-E001: CUDA_ERROR (GPU operation failed)
/// - EMB-E004: WEIGHT_CHECKSUM_MISMATCH (corrupted file)
/// - EMB-E005: DIMENSION_MISMATCH (wrong matrix shape)
/// - EMB-E006: PROJECTION_MATRIX_MISSING (file not found)
/// - EMB-E008: NOT_INITIALIZED (weights not loaded)
///
/// # Fail Fast Policy (Constitution AP-007)
/// All errors are non-recoverable. System MUST panic, NOT fall back to hash projection.
#[derive(Debug, Error)]
pub enum ProjectionError {
    /// Weight file not found at expected path.
    ///
    /// # Remediation
    /// Download from: https://huggingface.co/contextgraph/sparse-projection
    #[error("[EMB-E006] PROJECTION_MATRIX_MISSING: Weight file not found at {path}
  Expected: models/sparse_projection.safetensors
  Remediation: Download projection weights or train custom matrix")]
    MatrixMissing { path: PathBuf },

    /// Weight file checksum does not match expected value.
    #[error("[EMB-E004] WEIGHT_CHECKSUM_MISMATCH: Corrupted weight file
  Expected checksum: {expected}
  Actual checksum: {actual}
  File: {path}
  Remediation: Re-download weight file from trusted source")]
    ChecksumMismatch {
        path: PathBuf,
        expected: String,
        actual: String,
    },

    /// Weight matrix has wrong shape.
    #[error("[EMB-E005] DIMENSION_MISMATCH: Projection matrix has wrong shape
  Expected: [30522, 1536]
  Actual: [{actual_rows}, {actual_cols}]
  File: {path}
  Remediation: Ensure weight file matches BERT vocab (30522) to projection dim (1536)")]
    DimensionMismatch {
        path: PathBuf,
        actual_rows: usize,
        actual_cols: usize,
    },

    /// GPU operation failed during projection.
    #[error("[EMB-E001] CUDA_ERROR: GPU operation failed
  Operation: {operation}
  Details: {details}
  Remediation: Check GPU availability with nvidia-smi, verify driver version >= 545")]
    GpuError { operation: String, details: String },

    /// Projection weights not loaded (must call load() first).
    #[error("[EMB-E008] NOT_INITIALIZED: Projection weights not loaded
  Remediation: Call ProjectionMatrix::load() before calling project()")]
    NotInitialized,
}
```

---

## Files to Modify Summary

| File | Change |
|------|--------|
| `crates/context-graph-embeddings/src/models/pretrained/sparse/projection.rs` | Add `use std::path::PathBuf;`, `use thiserror::Error;`, and `ProjectionError` enum |
| `crates/context-graph-embeddings/src/models/pretrained/sparse/mod.rs` | Add `ProjectionError` to re-exports |

### mod.rs Change Required

In `crates/context-graph-embeddings/src/models/pretrained/sparse/mod.rs`, line 61 currently reads:
```rust
pub use projection::{ProjectionMatrix, PROJECTION_TENSOR_NAME, PROJECTION_WEIGHT_FILE};
```

Change to:
```rust
pub use projection::{ProjectionError, ProjectionMatrix, PROJECTION_TENSOR_NAME, PROJECTION_WEIGHT_FILE};
```

---

## Validation Criteria

- [ ] 5 error variants defined: `MatrixMissing`, `ChecksumMismatch`, `DimensionMismatch`, `GpuError`, `NotInitialized`
- [ ] Error codes match SPEC-EMB-001: EMB-E001, EMB-E004, EMB-E005, EMB-E006, EMB-E008
- [ ] Each error includes remediation steps in message
- [ ] `#[derive(Debug, Error)]` from thiserror
- [ ] `ProjectionError` exported from `sparse/mod.rs`
- [ ] `cargo check -p context-graph-embeddings` compiles without errors
- [ ] Tests pass: `cargo test -p context-graph-embeddings projection`

---

## Test Commands

Execute in order:

```bash
cd /home/cabdru/contextgraph

# 1. Verify enum does not exist yet (should return empty)
grep -n "ProjectionError" crates/context-graph-embeddings/src/models/pretrained/sparse/projection.rs

# 2. After implementation - verify enum exists
grep -n "pub enum ProjectionError" crates/context-graph-embeddings/src/models/pretrained/sparse/projection.rs
# Expected: line number with "pub enum ProjectionError"

# 3. Compile check
cargo check -p context-graph-embeddings 2>&1 | tee /tmp/check.log
# Expected: Compiling... Finished (no errors)

# 4. Run all projection tests
cargo test -p context-graph-embeddings projection -- --nocapture 2>&1 | tee /tmp/test.log
# Expected: 4+ tests pass (existing tests + any new ones)

# 5. Verify export in mod.rs
grep "ProjectionError" crates/context-graph-embeddings/src/models/pretrained/sparse/mod.rs
# Expected: "pub use projection::{ProjectionError, ..."

# 6. Verify error codes in output
grep -n "EMB-E" crates/context-graph-embeddings/src/models/pretrained/sparse/projection.rs
# Expected: Lines with EMB-E001, EMB-E004, EMB-E005, EMB-E006, EMB-E008
```

---

## Full State Verification Protocol

**CRITICAL: After completing the implementation, you MUST verify the system state through direct inspection.**

### Source of Truth Locations

| Data | Location | Expected State |
|------|----------|----------------|
| ProjectionError enum | `sparse/projection.rs` | 5 variants defined |
| Error codes | `sparse/projection.rs` | EMB-E001, EMB-E004, EMB-E005, EMB-E006, EMB-E008 |
| Module export | `sparse/mod.rs` | Contains `ProjectionError` in pub use |
| Imports | `sparse/projection.rs` | Has `use std::path::PathBuf;` and `use thiserror::Error;` |

### Execute & Inspect Pattern

**Step 1: Verify imports were added**
```bash
grep -n "use thiserror::Error" crates/context-graph-embeddings/src/models/pretrained/sparse/projection.rs
grep -n "use std::path::PathBuf" crates/context-graph-embeddings/src/models/pretrained/sparse/projection.rs
```
**Expected:** Two lines with the imports.

**Step 2: Verify enum definition**
```bash
grep -A 50 "pub enum ProjectionError" crates/context-graph-embeddings/src/models/pretrained/sparse/projection.rs | head -55
```
**Expected:** Full enum definition with 5 variants.

**Step 3: Verify each error variant exists**
```bash
grep -c "MatrixMissing" crates/context-graph-embeddings/src/models/pretrained/sparse/projection.rs
grep -c "ChecksumMismatch" crates/context-graph-embeddings/src/models/pretrained/sparse/projection.rs
grep -c "DimensionMismatch" crates/context-graph-embeddings/src/models/pretrained/sparse/projection.rs
grep -c "GpuError" crates/context-graph-embeddings/src/models/pretrained/sparse/projection.rs
grep -c "NotInitialized" crates/context-graph-embeddings/src/models/pretrained/sparse/projection.rs
```
**Expected:** Each returns "1" or more.

**Step 4: Verify export**
```bash
grep "ProjectionError" crates/context-graph-embeddings/src/models/pretrained/sparse/mod.rs
```
**Expected:** Line contains "pub use projection::{ProjectionError, ..."

**Step 5: Compile and test**
```bash
cargo check -p context-graph-embeddings && echo "CHECK PASSED" || echo "CHECK FAILED"
cargo test -p context-graph-embeddings projection 2>&1 | tail -10
```
**Expected:** "CHECK PASSED" and tests ok.

### Boundary & Edge Case Audit

**Edge Case 1: thiserror import missing**
- Before: No `use thiserror::Error;`
- After: Line exists near top of file with other imports
```bash
head -25 crates/context-graph-embeddings/src/models/pretrained/sparse/projection.rs | grep thiserror
```

**Edge Case 2: PathBuf import missing**
- Error messages use `PathBuf` type - must be imported
```bash
head -25 crates/context-graph-embeddings/src/models/pretrained/sparse/projection.rs | grep PathBuf
```

**Edge Case 3: Missing public export**
- Enum defined but not exported = unusable
```bash
grep "pub use projection.*ProjectionError" crates/context-graph-embeddings/src/models/pretrained/sparse/mod.rs
```

### Evidence of Success Log

After implementation, populate this log:

```
[VERIFICATION LOG - TASK-EMB-003]
Date: YYYY-MM-DD HH:MM

1. Imports Added:
   - use thiserror::Error: PRESENT/MISSING (line #)
   - use std::path::PathBuf: PRESENT/MISSING (line #)

2. Enum Definition:
   - pub enum ProjectionError: PRESENT/MISSING (line #)
   - #[derive(Debug, Error)]: PRESENT/MISSING

3. Variants (5 required):
   - MatrixMissing { path: PathBuf }: PRESENT/MISSING
   - ChecksumMismatch { path, expected, actual }: PRESENT/MISSING
   - DimensionMismatch { path, actual_rows, actual_cols }: PRESENT/MISSING
   - GpuError { operation, details }: PRESENT/MISSING
   - NotInitialized: PRESENT/MISSING

4. Error Codes in Messages:
   - [EMB-E001]: PRESENT/MISSING
   - [EMB-E004]: PRESENT/MISSING
   - [EMB-E005]: PRESENT/MISSING
   - [EMB-E006]: PRESENT/MISSING
   - [EMB-E008]: PRESENT/MISSING

5. Remediation in Messages:
   - MatrixMissing has "Download" instruction: YES/NO
   - ChecksumMismatch has "Re-download" instruction: YES/NO
   - DimensionMismatch has dimension info: YES/NO
   - GpuError has nvidia-smi instruction: YES/NO
   - NotInitialized has load() instruction: YES/NO

6. Module Export:
   - ProjectionError in mod.rs pub use: PRESENT/MISSING

7. Compilation:
   - cargo check -p context-graph-embeddings: SUCCESS/FAILED

8. Tests:
   - cargo test projection: X passed / Y failed
```

---

## What NOT to Do

| Action | Reason |
|--------|--------|
| Implement `From` conversions | Logic Layer task |
| Add error handling in `ProjectionMatrix` | Logic Layer task |
| Create separate error file | Keep with `ProjectionMatrix` for locality |
| Use `anyhow` instead of `thiserror` | Must match crate patterns |
| Add fallback behavior | Constitution AP-007 violation |
| Make errors recoverable | All projection errors are fatal |
| Use mock data in tests | Constitution AP-007 violation |

---

## Related Tasks

| Task | Relationship | Status |
|------|--------------|--------|
| TASK-EMB-001 | Prerequisite (dimensions) | ready |
| TASK-EMB-002 | Prerequisite (ProjectionMatrix) | **COMPLETE** |
| TASK-EMB-007 | Successor (unified error types) | pending |
| TASK-EMB-011 | Uses `ProjectionError` (load impl) | pending |
| TASK-EMB-012 | Uses `ProjectionError` (project impl) | pending |

---

## Error Code Reference (from SPEC-EMB-001)

| Code | Name | Used For |
|------|------|----------|
| EMB-E001 | CUDA_ERROR | GPU operation failure |
| EMB-E004 | WEIGHT_CHECKSUM_MISMATCH | Corrupted weight file |
| EMB-E005 | DIMENSION_MISMATCH | Wrong matrix shape |
| EMB-E006 | PROJECTION_MATRIX_MISSING | Weight file not found |
| EMB-E008 | NOT_INITIALIZED | Weights not loaded |

---

## Traceability

| Requirement | Tech Spec | Constitution | Code Location |
|-------------|-----------|--------------|---------------|
| REQ-EMB-001 | TECH-EMB-001 | E6_Sparse | `ProjectionError::DimensionMismatch` |
| - | - | AP-007 (fail fast) | All variants are non-recoverable |
| - | SPEC-EMB-001 Error Taxonomy | - | Error codes EMB-E001/4/5/6/8 |

</task_spec>
