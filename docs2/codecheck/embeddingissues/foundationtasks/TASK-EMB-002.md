# TASK-EMB-002: Create ProjectionMatrix Struct

<task_spec id="TASK-EMB-002" version="2.0" updated="2026-01-06">

## Metadata

| Field | Value |
|-------|-------|
| **Task ID** | TASK-EMB-002 |
| **Title** | Create ProjectionMatrix Struct |
| **Status** | COMPLETE (verified 2026-01-06) |
| **Layer** | foundation |
| **Sequence** | 2 |
| **Implements** | REQ-EMB-001 |
| **Depends On** | TASK-EMB-001 (VERIFIED COMPLETE - see Prerequisites) |
| **Estimated Complexity** | medium |
| **Constitution Ref** | `E6_Sparse: { dim: "~30K 5%active" }`, `AP-007` (No stub data in prod) |

---

## Problem Statement

**The current hash-based projection in `SparseVector::to_dense_projected()` destroys semantic information.**

Located in: `crates/context-graph-embeddings/src/models/pretrained/sparse/types.rs:71-91`

```rust
// CURRENT BROKEN CODE (lines 71-91)
pub fn to_dense_projected(&self, projected_dim: usize) -> Vec<f32> {
    let mut dense = vec![0.0f32; projected_dim];
    for (&idx, &weight) in self.indices.iter().zip(&self.weights) {
        let dense_idx = idx % projected_dim;  // DESTROYS SEMANTIC INFO
        dense[dense_idx] += weight;
    }
    // L2 normalize...
}
```

**Why This Is Catastrophic:**
1. Hash collisions destroy semantics: "machine" (token 3057) and "learning" (token 4083) map to same dense index if `3057 % 1536 == 4083 % 1536`
2. No learned representation - hash modulo is random noise masquerading as structure
3. **Violates Constitution AP-007**: "Stub data in prod" - this is effectively stub/mock behavior

**The Fix:** Create `ProjectionMatrix` struct that will later load a learned projection matrix (in Logic Layer tasks). This Foundation task creates the **data structure only**.

---

## Prerequisites Status (VERIFIED 2026-01-06)

### TASK-EMB-001: COMPLETE
The following have been verified in the codebase:

| File | Constant | Current Value | Status |
|------|----------|---------------|--------|
| `sparse/types.rs:35` | `SPARSE_PROJECTED_DIMENSION` | 1536 | ✅ Correct |
| `sparse/types.rs:38-41` | Compile-time assertion | Present | ✅ Correct |
| `dimensions/constants.rs:70` | `SPARSE` | 1536 | ✅ Correct |
| `tests.rs:148` | `test_embed_returns_1536d_vector` | 1536 | ✅ Correct |
| `tests.rs:224` | `test_sparse_vector_to_dense` | 1536 | ✅ Correct |

### Candle Dependency: AVAILABLE
```toml
# Cargo.toml lines 64-81
candle-core = { workspace = true, optional = true }
candle-nn = { workspace = true, optional = true }
default = ["candle"]  # MANDATORY for RTX 5090 acceleration
```

---

## Scope

### In Scope
- Create `ProjectionMatrix` struct in new file `sparse/projection.rs`
- Add struct fields: `weights: Tensor`, `device: Device`, `weight_checksum: [u8; 32]`
- Add associated constants for expected shape
- Add basic accessor methods (read-only)
- Export from `sparse/mod.rs`

### Out of Scope (Logic Layer Tasks)
- `ProjectionMatrix::load()` implementation → TASK-EMB-011
- `ProjectionMatrix::project()` implementation → TASK-EMB-012
- Modifying `SparseVector::to_dense_projected()` → TASK-EMB-012
- CUDA kernel integration → TASK-EMB-012
- Weight file training/creation → separate training task

---

## Files to Create

### File 1: `crates/context-graph-embeddings/src/models/pretrained/sparse/projection.rs`

```rust
//! Projection matrix for sparse-to-dense embedding conversion.
//!
//! This module defines the ProjectionMatrix struct for learned sparse projections.
//! The actual loading and projection logic are implemented in Logic Layer tasks.
//!
//! # Constitution Alignment
//! - E6_Sparse: "~30K 5%active" → 1536D output via learned projection
//! - E13_SPLADE: Same architecture, same projection
//! - AP-007: No stub data in prod - hash fallback is FORBIDDEN
//!
//! # CRITICAL: No Fallback Policy
//! If the weight file is missing or invalid, the system MUST fail fast with a clear
//! error message. Under NO circumstances should the code fall back to hash-based
//! projection (`idx % projected_dim`). Such fallback violates Constitution AP-007.

use candle_core::{Device, Tensor};

use super::types::{SPARSE_PROJECTED_DIMENSION, SPARSE_VOCAB_SIZE};

/// Expected weight file path relative to model directory.
pub const PROJECTION_WEIGHT_FILE: &str = "sparse_projection.safetensors";

/// Expected tensor name in SafeTensors file.
pub const PROJECTION_TENSOR_NAME: &str = "projection.weight";

/// Learned projection matrix for sparse-to-dense conversion.
///
/// # Constitution Alignment
/// - E6_Sparse: `dim: "~30K 5%active"` input, 1536D output
/// - E13_Splade: Same architecture, same projection
///
/// # Weight Source
/// - Pre-trained via contrastive learning on MS MARCO
/// - Fine-tuned to preserve semantic similarity
///
/// # CRITICAL: No Fallback
/// If weight file is missing, system MUST panic. Hash fallback is FORBIDDEN (AP-007).
///
/// # Memory Layout
/// - Weight tensor: [30522, 1536] float32 = ~180MB on GPU
/// - Total VRAM requirement: ~180MB for weights only
#[derive(Debug)]
pub struct ProjectionMatrix {
    /// Weight tensor on GPU: [SPARSE_VOCAB_SIZE x SPARSE_PROJECTED_DIMENSION]
    /// Shape: [30522, 1536]
    weights: Tensor,

    /// Device where weights are loaded (must be CUDA for production)
    device: Device,

    /// SHA256 checksum of the weight file for integrity validation
    weight_checksum: [u8; 32],
}

impl ProjectionMatrix {
    /// Expected weight matrix shape: [vocab_size, projected_dim]
    /// Shape: [30522, 1536] per Constitution E6_Sparse
    pub const EXPECTED_SHAPE: (usize, usize) = (SPARSE_VOCAB_SIZE, SPARSE_PROJECTED_DIMENSION);

    /// Expected file size in bytes: vocab_size * proj_dim * sizeof(f32)
    /// 30522 * 1536 * 4 = 187,527,168 bytes (~179MB)
    pub const EXPECTED_FILE_SIZE: usize = SPARSE_VOCAB_SIZE * SPARSE_PROJECTED_DIMENSION * 4;

    /// Get the weight tensor reference.
    ///
    /// # Returns
    /// Reference to the projection weight tensor [30522, 1536]
    #[inline]
    pub fn weights(&self) -> &Tensor {
        &self.weights
    }

    /// Get the device where weights are stored.
    ///
    /// # Returns
    /// Reference to the Candle Device (should be CUDA in production)
    #[inline]
    pub fn device(&self) -> &Device {
        &self.device
    }

    /// Get the weight file checksum for integrity verification.
    ///
    /// # Returns
    /// SHA256 checksum as 32-byte array
    #[inline]
    pub fn checksum(&self) -> &[u8; 32] {
        &self.weight_checksum
    }

    /// Check if weights are on a CUDA device.
    ///
    /// # Returns
    /// `true` if device is CUDA, `false` otherwise (e.g., CPU for testing)
    #[inline]
    pub fn is_cuda(&self) -> bool {
        matches!(self.device, Device::Cuda(_))
    }

    /// Get the input dimension (vocabulary size).
    ///
    /// # Returns
    /// 30522 (BERT vocabulary size)
    #[inline]
    pub const fn input_dimension() -> usize {
        SPARSE_VOCAB_SIZE
    }

    /// Get the output dimension (projected dimension).
    ///
    /// # Returns
    /// 1536 per Constitution E6_Sparse
    #[inline]
    pub const fn output_dimension() -> usize {
        SPARSE_PROJECTED_DIMENSION
    }
}

// Compile-time assertions to ensure constants match Constitution
const _: () = assert!(
    SPARSE_VOCAB_SIZE == 30522,
    "SPARSE_VOCAB_SIZE must be 30522 (BERT vocabulary)"
);

const _: () = assert!(
    SPARSE_PROJECTED_DIMENSION == 1536,
    "SPARSE_PROJECTED_DIMENSION must be 1536 per Constitution E6_Sparse"
);

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_expected_shape_constants() {
        assert_eq!(ProjectionMatrix::EXPECTED_SHAPE, (30522, 1536));
        assert_eq!(ProjectionMatrix::input_dimension(), 30522);
        assert_eq!(ProjectionMatrix::output_dimension(), 1536);
    }

    #[test]
    fn test_expected_file_size() {
        // 30522 * 1536 * 4 bytes = 187,527,168 bytes
        assert_eq!(ProjectionMatrix::EXPECTED_FILE_SIZE, 187_527_168);
        assert_eq!(
            ProjectionMatrix::EXPECTED_FILE_SIZE,
            30522 * 1536 * 4,
            "File size calculation must match vocab_size * proj_dim * sizeof(f32)"
        );
    }

    #[test]
    fn test_constants_match() {
        assert_eq!(
            ProjectionMatrix::EXPECTED_SHAPE.0,
            SPARSE_VOCAB_SIZE,
            "Expected shape row must match SPARSE_VOCAB_SIZE"
        );
        assert_eq!(
            ProjectionMatrix::EXPECTED_SHAPE.1,
            SPARSE_PROJECTED_DIMENSION,
            "Expected shape col must match SPARSE_PROJECTED_DIMENSION"
        );
    }

    #[test]
    fn test_weight_file_constants() {
        assert_eq!(PROJECTION_WEIGHT_FILE, "sparse_projection.safetensors");
        assert_eq!(PROJECTION_TENSOR_NAME, "projection.weight");
    }
}
```

---

## Files to Modify

### File: `crates/context-graph-embeddings/src/models/pretrained/sparse/mod.rs`

**Current content (lines 45-65):**
```rust
mod attention;
mod embeddings;
mod encoder;
mod forward;
mod loader;
mod mlm_head;
mod model;
mod traits;
mod types;

#[cfg(test)]
mod tests;

// Re-export used public types
pub use model::SparseModel;
#[allow(unused_imports)]
pub use types::{
    SparseVector, SPARSE_EXPECTED_SPARSITY, SPARSE_HIDDEN_SIZE, SPARSE_LATENCY_BUDGET_MS,
    SPARSE_MAX_TOKENS, SPARSE_MODEL_NAME, SPARSE_NATIVE_DIMENSION, SPARSE_PROJECTED_DIMENSION,
    SPARSE_VOCAB_SIZE,
};
```

**Add after line 53 (`mod types;`):**
```rust
mod projection;
```

**Add to re-exports (after line 65):**
```rust
pub use projection::{ProjectionMatrix, PROJECTION_TENSOR_NAME, PROJECTION_WEIGHT_FILE};
```

---

## Verification Commands

Execute in order. ALL must pass:

```bash
cd /home/cabdru/contextgraph

# 1. Verify file was created
ls -la crates/context-graph-embeddings/src/models/pretrained/sparse/projection.rs
# Expected: File exists with ~4KB size

# 2. Compile check (compile-time assertions catch mismatches)
cargo check -p context-graph-embeddings 2>&1 | tee /tmp/check.log
# Expected: Compiling context-graph-embeddings...Finished

# 3. Run unit tests in projection module
cargo test -p context-graph-embeddings projection::tests -- --nocapture 2>&1 | tee /tmp/test.log
# Expected: test result: ok. 4 passed; 0 failed

# 4. Generate documentation
cargo doc -p context-graph-embeddings --no-deps 2>&1 | grep -E "(Documenting|error)"
# Expected: Documenting context-graph-embeddings...

# 5. Verify struct is exported
grep -n "ProjectionMatrix" crates/context-graph-embeddings/src/models/pretrained/sparse/mod.rs
# Expected: Line with "pub use projection::ProjectionMatrix"

# 6. Verify no load/project implementations yet (should return empty)
grep -rn "fn load\|fn project" crates/context-graph-embeddings/src/models/pretrained/sparse/projection.rs | grep -v "test\|doc"
# Expected: Empty output (no implementations)
```

---

## Full State Verification Protocol

After completing the implementation, you MUST verify the system state through direct inspection.

### Source of Truth Locations

| Data | Location | Expected State |
|------|----------|----------------|
| ProjectionMatrix struct | `sparse/projection.rs` | Exists with 3 fields: weights, device, checksum |
| Module export | `sparse/mod.rs` | Contains `mod projection;` and `pub use projection::*` |
| Compile-time assertions | `sparse/projection.rs` | 2 assertions for vocab_size=30522 and proj_dim=1536 |
| Unit tests | `sparse/projection.rs::#[cfg(test)]` | 4 tests defined |

### Verification Steps (Execute & Inspect Pattern)

**Step 1: Verify File Exists**
```bash
cat crates/context-graph-embeddings/src/models/pretrained/sparse/projection.rs | head -30
```
**Expected Output:** First 30 lines of the module documentation and imports.

**Step 2: Verify Struct Definition**
```bash
grep -A 20 "pub struct ProjectionMatrix" crates/context-graph-embeddings/src/models/pretrained/sparse/projection.rs
```
**Expected Output:** Struct with `weights: Tensor`, `device: Device`, `weight_checksum: [u8; 32]`

**Step 3: Verify Compile-Time Assertions**
```bash
grep -n "const _: ()" crates/context-graph-embeddings/src/models/pretrained/sparse/projection.rs
```
**Expected Output:** Two lines with assertions for 30522 and 1536

**Step 4: Verify Module Export**
```bash
grep -n "projection" crates/context-graph-embeddings/src/models/pretrained/sparse/mod.rs
```
**Expected Output:**
- Line with `mod projection;`
- Line with `pub use projection::{ProjectionMatrix, ...}`

**Step 5: Run Tests and Capture Output**
```bash
cargo test -p context-graph-embeddings projection::tests -- --nocapture 2>&1
```
**Expected Output:**
```
running 4 tests
test models::pretrained::sparse::projection::tests::test_expected_shape_constants ... ok
test models::pretrained::sparse::projection::tests::test_expected_file_size ... ok
test models::pretrained::sparse::projection::tests::test_constants_match ... ok
test models::pretrained::sparse::projection::tests::test_weight_file_constants ... ok
test result: ok. 4 passed; 0 failed; 0 ignored
```

### Edge Cases to Manually Verify

**Edge Case 1: Empty projection.rs file**
- Before: File does not exist
- After: File exists with >100 lines
```bash
wc -l crates/context-graph-embeddings/src/models/pretrained/sparse/projection.rs
# Expected: ~120+ lines
```

**Edge Case 2: Missing mod declaration**
- Check mod.rs includes the module
```bash
grep "mod projection" crates/context-graph-embeddings/src/models/pretrained/sparse/mod.rs
# Expected: "mod projection;" present
```

**Edge Case 3: Missing public export**
- Check the type is publicly accessible
```bash
grep "pub use projection" crates/context-graph-embeddings/src/models/pretrained/sparse/mod.rs
# Expected: "pub use projection::{ProjectionMatrix, ...}" present
```

### Evidence of Success Log

After implementation, provide this log:
```
[VERIFICATION LOG - TASK-EMB-002]
Date: YYYY-MM-DD HH:MM

1. File Creation:
   - projection.rs exists: YES/NO
   - File size: XXX bytes
   - Line count: XXX lines

2. Struct Definition:
   - ProjectionMatrix struct: PRESENT/MISSING
   - weights: Tensor field: PRESENT/MISSING
   - device: Device field: PRESENT/MISSING
   - weight_checksum: [u8; 32] field: PRESENT/MISSING

3. Constants:
   - EXPECTED_SHAPE = (30522, 1536): VERIFIED/FAILED
   - EXPECTED_FILE_SIZE = 187,527,168: VERIFIED/FAILED
   - PROJECTION_WEIGHT_FILE = "sparse_projection.safetensors": VERIFIED/FAILED
   - PROJECTION_TENSOR_NAME = "projection.weight": VERIFIED/FAILED

4. Compile-Time Assertions:
   - SPARSE_VOCAB_SIZE == 30522 assertion: PRESENT/MISSING
   - SPARSE_PROJECTED_DIMENSION == 1536 assertion: PRESENT/MISSING

5. Module Integration:
   - mod projection; in mod.rs: PRESENT/MISSING
   - pub use projection::ProjectionMatrix: PRESENT/MISSING

6. Tests:
   - cargo check: SUCCESS/FAILED
   - cargo test projection::tests: X passed / Y failed

7. No Implementations (Foundation Layer Only):
   - fn load() exists: NO (correct for foundation)
   - fn project() exists: NO (correct for foundation)
```

---

## Definition of Done

### Exact Deliverables

- [ ] File `projection.rs` created at exact path: `crates/context-graph-embeddings/src/models/pretrained/sparse/projection.rs`
- [ ] `ProjectionMatrix` struct with 3 private fields
- [ ] 6 accessor methods: `weights()`, `device()`, `checksum()`, `is_cuda()`, `input_dimension()`, `output_dimension()`
- [ ] 2 associated constants: `EXPECTED_SHAPE`, `EXPECTED_FILE_SIZE`
- [ ] 2 module constants: `PROJECTION_WEIGHT_FILE`, `PROJECTION_TENSOR_NAME`
- [ ] 2 compile-time assertions for Constitution alignment
- [ ] 4 unit tests passing
- [ ] Module exported in `mod.rs`
- [ ] `cargo check -p context-graph-embeddings` succeeds
- [ ] `cargo test -p context-graph-embeddings projection::tests` passes
- [ ] NO `load()` or `project()` implementations (those are Logic Layer)

### Constraints

- Struct only, no method implementations beyond accessors
- Must use Candle types (`Tensor`, `Device`) from `candle_core`
- Must reference dimension constants from `super::types`
- NO backwards compatibility shims
- NO fallback implementations
- Fail fast with clear errors if anything is wrong

---

## What NOT to Do

| Action | Reason |
|--------|--------|
| Implement `ProjectionMatrix::load()` | Logic Layer task (TASK-EMB-011) |
| Implement `ProjectionMatrix::project()` | Logic Layer task (TASK-EMB-012) |
| Modify `SparseVector::to_dense_projected()` | Logic Layer task (TASK-EMB-012) |
| Add CUDA kernels | Logic Layer task |
| Create weight files | Separate training task |
| Add workarounds or fallbacks | Constitution AP-007 violation |
| Use mock data in tests | Constitution AP-007 violation |

---

## Traceability

| Requirement | Tech Spec | Constitution | Code Location |
|-------------|-----------|--------------|---------------|
| REQ-EMB-001 | TECH-EMB-001 | E6_Sparse dim | `ProjectionMatrix::EXPECTED_SHAPE` |
| REQ-EMB-002 | TECH-EMB-001 | AP-007 no stubs | No hash fallback |
| - | - | SPARSE_VOCAB_SIZE | `types.rs:15` = 30522 |
| - | - | SPARSE_PROJECTED_DIMENSION | `types.rs:35` = 1536 |

---

## Related Tasks

| Task | Relationship | Status |
|------|--------------|--------|
| TASK-EMB-001 | Prerequisite (dimension constants) | ✅ COMPLETE |
| TASK-EMB-011 | Successor (load implementation) | pending |
| TASK-EMB-012 | Successor (project implementation) | pending |

---

## Notes

- This struct will be populated by TASK-EMB-011 (load) and used by TASK-EMB-012 (project)
- The checksum field ensures weight file integrity
- CUDA device requirement is documented but not enforced until Logic Layer
- The `#[derive(Debug)]` allows logging but tensor contents are opaque

---

## VERIFICATION LOG - TASK-EMB-002

**Date:** 2026-01-06
**Verified By:** Claude Code

### 1. File Creation
| Check | Status | Details |
|-------|--------|---------|
| projection.rs exists | ✅ YES | `crates/context-graph-embeddings/src/models/pretrained/sparse/projection.rs` |
| File size | 5479 bytes | Within expected range |
| Line count | 171 lines | >100 lines as required |

### 2. Struct Definition
| Check | Status |
|-------|--------|
| ProjectionMatrix struct | ✅ PRESENT |
| `weights: Tensor` field | ✅ PRESENT |
| `device: Device` field | ✅ PRESENT |
| `weight_checksum: [u8; 32]` field | ✅ PRESENT |

### 3. Constants
| Constant | Expected | Actual | Status |
|----------|----------|--------|--------|
| `EXPECTED_SHAPE` | `(30522, 1536)` | `(30522, 1536)` | ✅ VERIFIED |
| `EXPECTED_FILE_SIZE` | `187,527,168` | `187,527,168` | ✅ VERIFIED |
| `PROJECTION_WEIGHT_FILE` | `"sparse_projection.safetensors"` | `"sparse_projection.safetensors"` | ✅ VERIFIED |
| `PROJECTION_TENSOR_NAME` | `"projection.weight"` | `"projection.weight"` | ✅ VERIFIED |

### 4. Compile-Time Assertions
| Assertion | Line | Status |
|-----------|------|--------|
| `SPARSE_VOCAB_SIZE == 30522` | 120 | ✅ PRESENT |
| `SPARSE_PROJECTED_DIMENSION == 1536` | 125 | ✅ PRESENT |

### 5. Module Integration
| Check | Status | Location |
|-------|--------|----------|
| `mod projection;` in mod.rs | ✅ PRESENT | Line 52 |
| `pub use projection::ProjectionMatrix` | ✅ PRESENT | Line 61 |
| `pub use projection::PROJECTION_TENSOR_NAME` | ✅ PRESENT | Line 61 |
| `pub use projection::PROJECTION_WEIGHT_FILE` | ✅ PRESENT | Line 61 |

### 6. Tests
| Command | Result |
|---------|--------|
| `cargo check -p context-graph-embeddings` | ✅ SUCCESS (warnings only, no errors) |
| `cargo test projection::tests` | ✅ 4 passed / 0 failed |
| `cargo doc -p context-graph-embeddings --no-deps` | ✅ SUCCESS |

### 7. Foundation Layer Compliance (No Implementations)
| Check | Status |
|-------|--------|
| `fn load()` exists | ✅ NO (correct for Foundation) |
| `fn project()` exists | ✅ NO (correct for Foundation) |

### 8. Edge Cases Verified
| Edge Case | Before State | After State | Result |
|-----------|--------------|-------------|--------|
| File existence | N/A (already created) | 171 lines, 5479 bytes | ✅ PASS |
| mod declaration | N/A | `mod projection;` at line 52 | ✅ PASS |
| Public export | N/A | `pub use projection::{...}` at line 61 | ✅ PASS |

### 9. Evidence of Success (Actual Test Output)
```
running 4 tests
test models::pretrained::sparse::projection::tests::test_constants_match ... ok
test models::pretrained::sparse::projection::tests::test_expected_file_size ... ok
test models::pretrained::sparse::projection::tests::test_expected_shape_constants ... ok
test models::pretrained::sparse::projection::tests::test_weight_file_constants ... ok

test result: ok. 4 passed; 0 failed; 0 ignored; 0 measured; 1171 filtered out
```

### Definition of Done Checklist
- [x] File `projection.rs` created at exact path
- [x] `ProjectionMatrix` struct with 3 private fields
- [x] 6 accessor methods: `weights()`, `device()`, `checksum()`, `is_cuda()`, `input_dimension()`, `output_dimension()`
- [x] 2 associated constants: `EXPECTED_SHAPE`, `EXPECTED_FILE_SIZE`
- [x] 2 module constants: `PROJECTION_WEIGHT_FILE`, `PROJECTION_TENSOR_NAME`
- [x] 2 compile-time assertions for Constitution alignment
- [x] 4 unit tests passing
- [x] Module exported in `mod.rs`
- [x] `cargo check -p context-graph-embeddings` succeeds
- [x] `cargo test -p context-graph-embeddings projection::tests` passes
- [x] NO `load()` or `project()` implementations (Foundation Layer only)

</task_spec>
