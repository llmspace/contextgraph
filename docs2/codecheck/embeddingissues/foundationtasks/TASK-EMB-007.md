# TASK-EMB-007: Enhance Consolidated Error Types with SPEC-EMB-001 Error Codes

<task_spec id="TASK-EMB-007" version="2.0">

## Metadata

| Field | Value |
|-------|-------|
| **Task ID** | TASK-EMB-007 |
| **Title** | Enhance EmbeddingError with SPEC-EMB-001 Error Taxonomy |
| **Status** | ready |
| **Layer** | foundation |
| **Sequence** | 7 |
| **Implements** | SPEC-EMB-001 Error Taxonomy (EMB-E001 through EMB-E012) |
| **Depends On** | TASK-EMB-005 (Storage types), TASK-EMB-006 (Warm loading types) |
| **Estimated Complexity** | medium |
| **Constitution Ref** | AP-007 "No Stub Data in Production", stack.gpu targeting RTX 5090 |

---

## ⚠️ CRITICAL REQUIREMENTS

### NO BACKWARDS COMPATIBILITY
- **FAIL FAST**: Invalid states MUST panic or return Err immediately
- **NO FALLBACKS**: Errors propagate; never silently handled
- **NO MOCK DATA**: All tests use real data structures from the crate
- **CONSTITUTION COMPLIANCE**: Must read `/home/cabdru/contextgraph/constitution.yaml`

### Required Constitution Reading
Before implementation, read and understand:
```yaml
# From constitution.yaml
principles:
  AP-007: "No Stub Data in Production"

stack:
  gpu:
    device: "NVIDIA RTX 5090 (Blackwell)"
    cuda: "13.1"
    compute_capability: "12.0"
    vram: "32GB GDDR7"

security:
  checksums:
    algorithm: "SHA256"
```

---

## Current Codebase State

### ✅ EXISTING: EmbeddingError at `crates/context-graph-embeddings/src/error/types.rs`

The crate already has `EmbeddingError` with these variants:
- `ModelNotFound`, `ModelLoadError`, `NotInitialized`, `ModelAlreadyLoaded`, `ModelNotLoaded`
- `MemoryBudgetExceeded`, `InternalError`
- `InvalidDimension`, `InvalidValue`, `EmptyInput`, `InputTooLong`, `InvalidImage`
- `BatchError`, `TokenizationError`
- `GpuError`, `CacheError`, `IoError`, `Timeout`
- `UnsupportedModality`, `ConfigError`
- `SerializationError`, `DimensionMismatch`

### ❌ MISSING: SPEC-EMB-001 Error Taxonomy Codes

The SPEC-EMB-001 defines 12 specific error codes (EMB-E001 through EMB-E012) that are NOT present:
- EMB-E001: CUDA_UNAVAILABLE
- EMB-E002: INSUFFICIENT_VRAM
- EMB-E003: WEIGHT_FILE_MISSING
- EMB-E004: WEIGHT_CHECKSUM_MISMATCH
- EMB-E005: DIMENSION_MISMATCH (exists as `InvalidDimension`, needs code)
- EMB-E006: PROJECTION_MATRIX_MISSING
- EMB-E007: OOM_DURING_BATCH
- EMB-E008: INFERENCE_VALIDATION_FAILED
- EMB-E009: INPUT_TOO_LARGE (exists as `InputTooLong`, needs code)
- EMB-E010: STORAGE_CORRUPTION
- EMB-E011: CODEBOOK_MISSING
- EMB-E012: RECALL_LOSS_EXCEEDED

### ❌ NOT FOUND: ProjectionError
- `grep -r "ProjectionError" crates/` returns NO matches
- The sparse projection module does NOT exist at `crates/context-graph-embeddings/src/models/pretrained/sparse/projection.rs`
- This task document previously referenced a non-existent file

### File Structure Reality
```
crates/context-graph-embeddings/src/
├── error/
│   ├── mod.rs          # Re-exports EmbeddingError, EmbeddingResult
│   └── types.rs        # EmbeddingError enum definition (134 lines)
├── quantization/
│   ├── mod.rs          # Exports QuantizationMethod, QuantizedEmbedding, etc.
│   └── types.rs        # QuantizationMethod::for_model_id() covers all 13 ModelIds
├── storage/
│   ├── mod.rs          # Re-exports storage types
│   └── types.rs        # StoredQuantizedFingerprint, IndexEntry, etc.
├── warm/
│   └── loader/
│       ├── mod.rs      # Re-exports loader types
│       └── types.rs    # TensorMetadata, WarmLoadResult, LoadedModelWeights
├── types/
│   └── model_id.rs     # ModelId enum with 13 variants
└── lib.rs              # Main exports (already exports error module)
```

---

## Scope

### In Scope
1. **ADD** new error variants for SPEC-EMB-001 codes to existing `EmbeddingError`
2. **ADD** `code() -> &'static str` method returning EMB-E001 through EMB-E012
3. **ADD** `is_recoverable() -> bool` method (only EMB-E009 is recoverable)
4. **ADD** remediation guidance in error messages
5. **ADD** comprehensive tests with REAL data (no mocks)

### Out of Scope
- Creating ProjectionError (does not exist, remove references)
- Error recovery logic (Logic Layer task)
- Changing existing variant semantics (additive only)

---

## Definition of Done

### 1. Enhanced Error Variants

Add to `crates/context-graph-embeddings/src/error/types.rs`:

```rust
use std::path::PathBuf;
use crate::types::ModelId;

// ADD these variants to existing EmbeddingError enum:

/// EMB-E001: CUDA is required but unavailable.
/// Constitution: stack.gpu.cuda = "13.1"
#[error("[EMB-E001] CUDA_UNAVAILABLE: {message}
  Required: RTX 5090 (Blackwell, CC 12.0), CUDA 13.1+
  Remediation: Install CUDA 13.1+ and verify GPU with nvidia-smi")]
CudaUnavailable { message: String },

/// EMB-E002: Insufficient GPU VRAM.
/// Constitution: stack.gpu.vram = "32GB GDDR7"
#[error("[EMB-E002] INSUFFICIENT_VRAM: GPU memory insufficient
  Required: {required_bytes} bytes ({required_gb:.1} GB)
  Available: {available_bytes} bytes ({available_gb:.1} GB)
  Remediation: Free GPU memory or upgrade to RTX 5090 (32GB)")]
InsufficientVram {
    required_bytes: usize,
    available_bytes: usize,
    required_gb: f64,
    available_gb: f64,
},

/// EMB-E003: Weight file not found.
#[error("[EMB-E003] WEIGHT_FILE_MISSING: Model weights not found
  Model: {model_id:?}
  Path: {path:?}
  Remediation: Download weights from HuggingFace model repository")]
WeightFileMissing { model_id: ModelId, path: PathBuf },

/// EMB-E004: Weight file checksum mismatch.
/// Constitution: security.checksums.algorithm = "SHA256"
#[error("[EMB-E004] WEIGHT_CHECKSUM_MISMATCH: Weight file corrupted
  Model: {model_id:?}
  Expected SHA256: {expected}
  Actual SHA256: {actual}
  Remediation: Re-download weight file from source")]
WeightChecksumMismatch {
    model_id: ModelId,
    expected: String,
    actual: String,
},

/// EMB-E005: Dimension mismatch during validation.
#[error("[EMB-E005] DIMENSION_MISMATCH: Embedding dimension invalid
  Model: {model_id:?}
  Expected: {expected}
  Actual: {actual}
  Remediation: Verify model configuration matches ModelId::dimension()")]
ModelDimensionMismatch {
    model_id: ModelId,
    expected: usize,
    actual: usize,
},

/// EMB-E006: Projection matrix file missing.
#[error("[EMB-E006] PROJECTION_MATRIX_MISSING: Sparse projection weights not found
  Path: {path:?}
  Remediation: Download from model repository or regenerate projection matrix")]
ProjectionMatrixMissing { path: PathBuf },

/// EMB-E007: Out of memory during batch processing.
#[error("[EMB-E007] OOM_DURING_BATCH: GPU OOM during batch inference
  Batch size: {batch_size}
  Model: {model_id:?}
  Remediation: Reduce batch size or free GPU memory")]
OomDuringBatch { batch_size: usize, model_id: ModelId },

/// EMB-E008: Inference validation failed (NaN, Inf, or zero-norm).
#[error("[EMB-E008] INFERENCE_VALIDATION_FAILED: Model output invalid
  Model: {model_id:?}
  Reason: {reason}
  Remediation: Verify model weights and input preprocessing")]
InferenceValidationFailed { model_id: ModelId, reason: String },

/// EMB-E009: Input exceeds model capacity (ONLY recoverable error).
#[error("[EMB-E009] INPUT_TOO_LARGE: Input exceeds token limit
  Max tokens: {max_tokens}
  Actual tokens: {actual_tokens}
  Remediation: Truncate input or split into chunks")]
InputTooLarge { max_tokens: usize, actual_tokens: usize },

/// EMB-E010: Storage corruption detected.
#[error("[EMB-E010] STORAGE_CORRUPTION: Stored embedding data corrupted
  Fingerprint ID: {id}
  Reason: {reason}
  Remediation: Re-index from source document")]
StorageCorruption { id: String, reason: String },

/// EMB-E011: PQ-8 codebook missing for quantization.
#[error("[EMB-E011] CODEBOOK_MISSING: PQ-8 codebook not found
  Model: {model_id:?}
  Remediation: Train codebook with representative vectors or download pre-trained")]
CodebookMissing { model_id: ModelId },

/// EMB-E012: Quantization recall loss exceeded threshold.
#[error("[EMB-E012] RECALL_LOSS_EXCEEDED: Quantization quality too low
  Model: {model_id:?}
  Measured recall: {measured:.4}
  Max allowed loss: {max_allowed:.4}
  Remediation: Retrain codebook with more representative data")]
RecallLossExceeded {
    model_id: ModelId,
    measured: f32,
    max_allowed: f32,
},
```

### 2. Required Methods

Add to `impl EmbeddingError`:

```rust
impl EmbeddingError {
    /// Returns the SPEC-EMB-001 error code if applicable.
    ///
    /// Returns None for variants that predate the specification.
    pub fn spec_code(&self) -> Option<&'static str> {
        match self {
            Self::CudaUnavailable { .. } => Some("EMB-E001"),
            Self::InsufficientVram { .. } => Some("EMB-E002"),
            Self::WeightFileMissing { .. } => Some("EMB-E003"),
            Self::WeightChecksumMismatch { .. } => Some("EMB-E004"),
            Self::ModelDimensionMismatch { .. } => Some("EMB-E005"),
            Self::ProjectionMatrixMissing { .. } => Some("EMB-E006"),
            Self::OomDuringBatch { .. } => Some("EMB-E007"),
            Self::InferenceValidationFailed { .. } => Some("EMB-E008"),
            Self::InputTooLarge { .. } => Some("EMB-E009"),
            Self::StorageCorruption { .. } => Some("EMB-E010"),
            Self::CodebookMissing { .. } => Some("EMB-E011"),
            Self::RecallLossExceeded { .. } => Some("EMB-E012"),
            // Legacy variants have no spec code
            _ => None,
        }
    }

    /// Check if this error is recoverable.
    ///
    /// Per SPEC-EMB-001, ONLY EMB-E009 (INPUT_TOO_LARGE) is recoverable.
    /// All other errors require operator intervention.
    pub fn is_recoverable(&self) -> bool {
        matches!(self, Self::InputTooLarge { .. } | Self::InputTooLong { .. })
    }

    /// Returns severity level for monitoring/alerting.
    pub fn severity(&self) -> ErrorSeverity {
        match self {
            // Critical: System cannot function
            Self::CudaUnavailable { .. }
            | Self::InsufficientVram { .. }
            | Self::WeightFileMissing { .. }
            | Self::WeightChecksumMismatch { .. }
            | Self::ModelDimensionMismatch { .. }
            | Self::ProjectionMatrixMissing { .. }
            | Self::InferenceValidationFailed { .. } => ErrorSeverity::Critical,

            // High: Operation failed, retry unlikely to help
            | Self::OomDuringBatch { .. }
            | Self::StorageCorruption { .. }
            | Self::CodebookMissing { .. } => ErrorSeverity::High,

            // Medium: Operation failed, may succeed with modification
            | Self::InputTooLarge { .. }
            | Self::RecallLossExceeded { .. } => ErrorSeverity::Medium,

            // Legacy variants default to High
            _ => ErrorSeverity::High,
        }
    }
}

/// Error severity for monitoring integration.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ErrorSeverity {
    Critical,
    High,
    Medium,
}
```

---

## Full State Verification Protocol

### 1. Source of Truth Verification

Before implementation, verify these files exist and match expected state:

| File | Expected State | Verification Command |
|------|----------------|---------------------|
| `crates/context-graph-embeddings/src/error/types.rs` | Contains `EmbeddingError` enum | `grep "pub enum EmbeddingError" crates/context-graph-embeddings/src/error/types.rs` |
| `crates/context-graph-embeddings/src/error/mod.rs` | Re-exports `EmbeddingError` | `grep "pub use types::EmbeddingError" crates/context-graph-embeddings/src/error/mod.rs` |
| `crates/context-graph-embeddings/src/types/model_id.rs` | Contains 13 `ModelId` variants | `grep -c "^    [A-Z]" crates/context-graph-embeddings/src/types/model_id.rs` |
| `constitution.yaml` | Contains AP-007, stack.gpu | `grep "AP-007\|RTX 5090" constitution.yaml` |

### 2. Execute & Inspect

After adding each error variant, verify:

```bash
# Trigger: Add new variant
# Process: Cargo type-checks the enum
# Outcome: Variant is accessible from lib.rs

# Step 1: Compile check
cargo check -p context-graph-embeddings 2>&1 | tee /tmp/emb007-compile.log
grep -q "error\[E" /tmp/emb007-compile.log && echo "FAIL: Compilation errors" && exit 1

# Step 2: Verify variant exists in public API
cargo doc -p context-graph-embeddings --no-deps 2>&1 | grep -q "CudaUnavailable"
echo "Variant exported: $?"

# Step 3: Run error tests
cargo test -p context-graph-embeddings error:: --no-fail-fast 2>&1 | tee /tmp/emb007-test.log
```

### 3. Edge Case Audit

| Edge Case | Test Scenario | Expected Behavior |
|-----------|---------------|-------------------|
| Zero VRAM available | `InsufficientVram { available_bytes: 0, .. }` | Error message shows "0.0 GB" |
| Empty path | `WeightFileMissing { path: PathBuf::new(), .. }` | Error displays empty path (no panic) |
| NaN in recall | `RecallLossExceeded { measured: f32::NAN, .. }` | Display shows "NaN" |
| Unicode model path | Path with emoji/CJK chars | Path displays correctly |
| Max u64 timeout | `Timeout { timeout_ms: u64::MAX }` | No overflow in display |

### 4. Evidence of Success

After implementation, log must show:

```
[TASK-EMB-007] Implementation Complete
├── Trigger: Task execution started
├── Process: Added 12 SPEC-EMB-001 error variants
├── Outcome: All variants compile and test
│
├── Verification Results:
│   ├── [✓] cargo check -p context-graph-embeddings: SUCCESS
│   ├── [✓] cargo test -p context-graph-embeddings error:: : 12/12 passed
│   ├── [✓] spec_code() returns correct codes for all 12 variants
│   ├── [✓] is_recoverable() returns true ONLY for InputTooLarge/InputTooLong
│   ├── [✓] severity() returns Critical/High/Medium appropriately
│   ├── [✓] All error messages include remediation steps
│   └── [✓] Constitution references accurate (RTX 5090, CUDA 13.1, SHA256)
│
└── Files Modified:
    ├── crates/context-graph-embeddings/src/error/types.rs (added variants + methods)
    └── crates/context-graph-embeddings/src/error/mod.rs (re-export ErrorSeverity)
```

---

## Manual Verification Checklist

After implementation, manually verify:

- [ ] `cargo check -p context-graph-embeddings` compiles without warnings
- [ ] `cargo test -p context-graph-embeddings error::` passes all tests
- [ ] `cargo doc -p context-graph-embeddings` generates docs for all new variants
- [ ] Error messages contain RTX 5090, CUDA 13.1, SHA256 where appropriate
- [ ] `is_recoverable()` returns `true` ONLY for `InputTooLarge` and `InputTooLong`
- [ ] `spec_code()` returns `Some("EMB-E001")` through `Some("EMB-E012")` for spec variants
- [ ] `spec_code()` returns `None` for legacy variants (ModelNotFound, etc.)

---

## Test Requirements (NO MOCK DATA)

### Test File Location
`crates/context-graph-embeddings/src/error/tests.rs`

### Required Tests

```rust
#[cfg(test)]
mod tests {
    use super::*;
    use std::path::PathBuf;

    // Use REAL ModelId variants - no mocks
    const TEST_MODEL: ModelId = ModelId::Semantic;

    #[test]
    fn test_cuda_unavailable_code() {
        let err = EmbeddingError::CudaUnavailable {
            message: "Driver not found".to_string(),
        };
        assert_eq!(err.spec_code(), Some("EMB-E001"));
        assert!(!err.is_recoverable());
        assert_eq!(err.severity(), ErrorSeverity::Critical);
        assert!(err.to_string().contains("EMB-E001"));
        assert!(err.to_string().contains("RTX 5090"));
    }

    #[test]
    fn test_insufficient_vram_code() {
        let err = EmbeddingError::InsufficientVram {
            required_bytes: 32_000_000_000,
            available_bytes: 8_000_000_000,
            required_gb: 32.0,
            available_gb: 8.0,
        };
        assert_eq!(err.spec_code(), Some("EMB-E002"));
        assert!(!err.is_recoverable());
        assert_eq!(err.severity(), ErrorSeverity::Critical);
    }

    #[test]
    fn test_weight_file_missing_code() {
        let err = EmbeddingError::WeightFileMissing {
            model_id: TEST_MODEL,
            path: PathBuf::from("/models/semantic/weights.safetensors"),
        };
        assert_eq!(err.spec_code(), Some("EMB-E003"));
        assert!(!err.is_recoverable());
    }

    #[test]
    fn test_weight_checksum_mismatch_code() {
        let err = EmbeddingError::WeightChecksumMismatch {
            model_id: TEST_MODEL,
            expected: "abc123".to_string(),
            actual: "def456".to_string(),
        };
        assert_eq!(err.spec_code(), Some("EMB-E004"));
        assert!(err.to_string().contains("SHA256"));
    }

    #[test]
    fn test_input_too_large_is_recoverable() {
        let err = EmbeddingError::InputTooLarge {
            max_tokens: 512,
            actual_tokens: 1024,
        };
        assert_eq!(err.spec_code(), Some("EMB-E009"));
        assert!(err.is_recoverable()); // ONLY recoverable error
        assert_eq!(err.severity(), ErrorSeverity::Medium);
    }

    #[test]
    fn test_legacy_variant_no_spec_code() {
        let err = EmbeddingError::ModelNotFound {
            model_id: TEST_MODEL,
        };
        assert_eq!(err.spec_code(), None); // Legacy = no spec code
        assert!(!err.is_recoverable());
    }

    #[test]
    fn test_all_spec_codes_unique() {
        let codes = [
            EmbeddingError::CudaUnavailable { message: "".into() }.spec_code(),
            EmbeddingError::InsufficientVram {
                required_bytes: 0, available_bytes: 0, required_gb: 0.0, available_gb: 0.0
            }.spec_code(),
            // ... all 12 variants
        ];
        let unique: std::collections::HashSet<_> = codes.iter().flatten().collect();
        assert_eq!(unique.len(), 12, "All 12 spec codes must be unique");
    }

    // Edge case: zero available VRAM
    #[test]
    fn test_zero_vram_display() {
        let err = EmbeddingError::InsufficientVram {
            required_bytes: 1000,
            available_bytes: 0,
            required_gb: 0.001,
            available_gb: 0.0,
        };
        let msg = err.to_string();
        assert!(msg.contains("0.0 GB") || msg.contains("0 bytes"));
    }

    // Edge case: NaN recall
    #[test]
    fn test_nan_recall_display() {
        let err = EmbeddingError::RecallLossExceeded {
            model_id: TEST_MODEL,
            measured: f32::NAN,
            max_allowed: 0.05,
        };
        let msg = err.to_string();
        // Should not panic, should display NaN
        assert!(msg.contains("NaN") || msg.contains("nan"));
    }
}
```

---

## Files to Modify

| File | Change |
|------|--------|
| `crates/context-graph-embeddings/src/error/types.rs` | Add 12 SPEC variants + methods + ErrorSeverity |
| `crates/context-graph-embeddings/src/error/mod.rs` | Re-export `ErrorSeverity` |

### Files NOT to Create
- No `projection.rs` - sparse projection module does not exist
- No new error modules - use existing `error/types.rs`

---

## Test Commands

```bash
cd /home/cabdru/contextgraph

# Verify prerequisites
cargo check -p context-graph-embeddings

# Run error tests specifically
cargo test -p context-graph-embeddings error:: --no-fail-fast -- --nocapture

# Verify all 12 codes are documented
cargo doc -p context-graph-embeddings --no-deps --open

# Full test suite
cargo test -p context-graph-embeddings
```

---

## Error Code Reference

| Code | Name | Recoverable | Severity | Constitution Ref |
|------|------|-------------|----------|------------------|
| EMB-E001 | CUDA_UNAVAILABLE | No | Critical | stack.gpu.cuda |
| EMB-E002 | INSUFFICIENT_VRAM | No | Critical | stack.gpu.vram |
| EMB-E003 | WEIGHT_FILE_MISSING | No | Critical | - |
| EMB-E004 | WEIGHT_CHECKSUM_MISMATCH | No | Critical | security.checksums |
| EMB-E005 | DIMENSION_MISMATCH | No | Critical | - |
| EMB-E006 | PROJECTION_MATRIX_MISSING | No | Critical | - |
| EMB-E007 | OOM_DURING_BATCH | No | High | - |
| EMB-E008 | INFERENCE_VALIDATION_FAILED | No | Critical | AP-007 |
| EMB-E009 | INPUT_TOO_LARGE | **Yes** | Medium | - |
| EMB-E010 | STORAGE_CORRUPTION | No | High | - |
| EMB-E011 | CODEBOOK_MISSING | No | High | - |
| EMB-E012 | RECALL_LOSS_EXCEEDED | No | Medium | - |

---

## Traceability

| Requirement | Tech Spec | Constitution | Issue |
|-------------|-----------|--------------|-------|
| EMB-E001-E012 | SPEC-EMB-001 Error Taxonomy | AP-007 | embedding-errors |
| GPU Requirements | SPEC-EMB-001 | stack.gpu | gpu-architecture |
| Checksum Validation | SPEC-EMB-001 | security.checksums | integrity |

</task_spec>
