# TASK-P2-006: SemanticFingerprint Quantizer Implementation

```xml
<task_spec id="TASK-P2-006" version="2.0">
<metadata>
  <title>SemanticFingerprint Quantizer Implementation</title>
  <status>COMPLETE</status>
  <layer>logic</layer>
  <sequence>19</sequence>
  <phase>2</phase>
  <implements>
    <requirement_ref>REQ-P2-03</requirement_ref>
  </implements>
  <depends_on>
    <task_ref>TASK-P2-005</task_ref>
  </depends_on>
  <estimated_complexity>high</estimated_complexity>
  <last_audited>2026-01-17</last_audited>
  <completed_date>2026-01-16</completed_date>
</metadata>
```

---

## CRITICAL AUDIT: Codebase Discrepancies (2026-01-17)

**The original task document v1.0 was WRONG.** This section documents verified actual state:

### File Path Corrections (VERIFIED)

| WRONG (v1.0) | CORRECT (Actual) |
|--------------|------------------|
| `src/embedding/` | `src/embeddings/` (plural!) |
| `src/embedding/teleological.rs` | `src/types/fingerprint/semantic/fingerprint.rs` |
| `src/embedding/vector.rs` | `src/embeddings/vector.rs` |
| SparseVector in `vector.rs` | `src/types/fingerprint/sparse.rs` |
| NEW `src/embedding/quantize.rs` | ADD TO `src/quantization/fingerprint.rs` |

### Type Corrections (VERIFIED)

| WRONG (v1.0) | CORRECT (Actual) |
|--------------|------------------|
| `TeleologicalArray` struct | `SemanticFingerprint` (alias: `pub type TeleologicalArray = SemanticFingerprint`) |
| `DenseVector<N>` generic | `DenseVector` with `Vec<f32>` (runtime-sized) |
| `SparseVector::data()` returns tuple | `.indices: Vec<u16>`, `.values: Vec<f32>` (direct fields) |
| `SparseVector::dimension()` | NO SUCH METHOD - use `SPARSE_VOCAB_SIZE = 30_522` |
| `e9_hdc: BinaryVector` | `e9_hdc: Vec<f32>` (1024D PROJECTED dense!) |
| `InvertedIndex.indices: Vec<u32>` | Use `Vec<u16>` (matches SparseVector) |

### Existing Infrastructure (VERIFIED)

The `quantization` module EXISTS at `crates/context-graph-core/src/quantization/`:

```
quantization/
├── mod.rs       # Exports: Quantizable, Precision, QuantizedEmbedding
├── types.rs     # Precision (Int4, Int8, Fp16), QuantizedEmbedding, QuantizationError
├── traits.rs    # Quantizable trait
├── batch.rs     # batch_quantize, batch_dequantize (rayon parallel)
├── accuracy.rs  # compute_rmse, compute_nrmse, AccuracyReport
```

**`TokenPruningEmbedding` (E12) already implements `Quantizable`** in `src/embeddings/token_pruning.rs`.

### Advanced PQ-8 Infrastructure (DISCOVERED)

A full-featured PQ-8 implementation exists at `crates/context-graph-embeddings/src/quantization/pq8/`:

```
pq8/
├── mod.rs       # PQ8Encoder exports
├── types.rs     # PQ8QuantizationError, KMeansConfig, SimpleRng
├── encoder.rs   # PQ8Encoder with trained codebooks, ADC distance
├── codebook.rs  # PQ8Codebook training via k-means++
```

**Key types from `types.rs`:**
- `NUM_SUBVECTORS = 8`, `NUM_CENTROIDS = 256`
- `PQ8QuantizationError` with comprehensive error variants
- `KMeansConfig` with `max_iterations`, `convergence_threshold`, `seed`

**This task uses simplified mean-based PQ** (no codebook training) for faster development. The advanced encoder can be integrated later for better accuracy.

---

## Context

Implements `QuantizedSemanticFingerprint` that compresses `SemanticFingerprint` from ~46KB to ~11KB:

| Method | Embedders | Description | Compression |
|--------|-----------|-------------|-------------|
| **PQ-8** | E1, E5, E7, E10 | Product quantization, 8-bit codes | 1024D -> 32B (32x) |
| **Float8** | E2-E4, E8, E9, E11 | Min-max scalar quantization | 1:4 ratio |
| **Token Float8** | E12 | Per-token Float8 | ~1.3KB for 10 tokens |
| **Sparse** | E6, E13 | u16 indices + Float8 values | ~2.5KB each |

**CRITICAL**: E9 is `Vec<f32>` (projected dense), NOT `BinaryVector`. The 10K-bit HDC vector is projected to 1024D before storage.

---

## Input Context Files

```
crates/context-graph-core/src/types/fingerprint/semantic/fingerprint.rs  # SemanticFingerprint
crates/context-graph-core/src/types/fingerprint/sparse.rs               # SparseVector
crates/context-graph-core/src/embeddings/vector.rs                       # DenseVector, BinaryVector
crates/context-graph-core/src/quantization/                              # Existing module
crates/context-graph-core/src/embeddings/token_pruning.rs                # E12 Quantizable impl
```

## Prerequisites (ALL VERIFIED COMPLETE)

- [x] `SemanticFingerprint` at `src/types/fingerprint/semantic/fingerprint.rs`
- [x] `SparseVector` at `src/types/fingerprint/sparse.rs`
- [x] `DenseVector`, `BinaryVector` at `src/embeddings/vector.rs`
- [x] `StubMultiArrayProvider` at `src/embeddings/provider.rs`
- [x] `Quantizable` trait at `src/quantization/traits.rs`
- [x] `TokenPruningEmbedding` implements `Quantizable`

---

## Scope

### In Scope

1. Create `QuantizedSemanticFingerprint` struct
2. Implement `quantize_fingerprint()` function
3. Implement `dequantize_fingerprint()` function
4. Float8 (scalar min-max) quantization helpers
5. PQ-8 (simplified mean-based) quantization helpers
6. Sparse quantization (preserve u16 indices, Float8 values)
7. `FingerprintQuantizeError` error type

### Out of Scope

- Trained PQ codebooks (use mean-based approximation)
- SIMD/GPU optimization
- Adaptive quantization

---

## Definition of Done

### Create: `crates/context-graph-core/src/quantization/fingerprint.rs`

```rust
//! SemanticFingerprint quantization for storage compression.
//! TASK-P2-006: ~46KB -> ~11KB via mixed quantization strategies.

use serde::{Deserialize, Serialize};
use thiserror::Error;
use crate::types::fingerprint::{SemanticFingerprint, SparseVector};
use crate::teleological::Embedder;

// ============================================================================
// Error Types
// ============================================================================

#[derive(Debug, Error, Clone)]
pub enum FingerprintQuantizeError {
    #[error("Invalid input for {embedder:?}: {message}")]
    InvalidInput { embedder: Embedder, message: String },

    #[error("Dimension mismatch for {embedder:?}: expected {expected}, got {actual}")]
    DimensionMismatch { embedder: Embedder, expected: usize, actual: usize },

    #[error("NaN/Infinity in {embedder:?} at index {index}")]
    InvalidValue { embedder: Embedder, index: usize },
}

// ============================================================================
// Quantized Types
// ============================================================================

/// Float8 quantized dense vector (min-max scalar)
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct QuantizedFloat8 {
    pub data: Vec<u8>,
    pub min_val: f32,
    pub max_val: f32,
    pub original_dim: usize,
}

/// PQ-8 quantized dense vector (simplified, no learned codebooks)
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct QuantizedPQ8 {
    pub codes: Vec<u8>,
    pub num_subvectors: usize,
    pub original_dim: usize,
}

/// Sparse with quantized values (indices exact, values Float8)
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct QuantizedSparse {
    pub indices: Vec<u16>,  // Preserved exactly
    pub quantized_values: Vec<u8>,
    pub min_val: f32,
    pub max_val: f32,
}

/// Complete quantized fingerprint (~11KB target)
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct QuantizedSemanticFingerprint {
    // PQ-8: E1, E5, E7, E10
    pub e1_semantic: QuantizedPQ8,
    pub e5_causal: QuantizedPQ8,
    pub e7_code: QuantizedPQ8,
    pub e10_multimodal: QuantizedPQ8,

    // Float8: E2, E3, E4, E8, E9, E11
    pub e2_temporal_recent: QuantizedFloat8,
    pub e3_temporal_periodic: QuantizedFloat8,
    pub e4_temporal_positional: QuantizedFloat8,
    pub e8_graph: QuantizedFloat8,
    pub e9_hdc: QuantizedFloat8,  // NOT binary - projected dense!
    pub e11_entity: QuantizedFloat8,

    // Token-level Float8: E12
    pub e12_late_interaction: Vec<QuantizedFloat8>,

    // Sparse: E6, E13
    pub e6_sparse: QuantizedSparse,
    pub e13_splade: QuantizedSparse,
}

// ============================================================================
// Public API
// ============================================================================

pub fn quantize_fingerprint(
    fp: &SemanticFingerprint
) -> Result<QuantizedSemanticFingerprint, FingerprintQuantizeError>;

pub fn dequantize_fingerprint(
    qfp: &QuantizedSemanticFingerprint
) -> Result<SemanticFingerprint, FingerprintQuantizeError>;

impl QuantizedSemanticFingerprint {
    pub fn estimated_size_bytes(&self) -> usize;
}
```

### Modify: `crates/context-graph-core/src/quantization/mod.rs`

Add:
```rust
pub mod fingerprint;

pub use fingerprint::{
    quantize_fingerprint, dequantize_fingerprint,
    QuantizedSemanticFingerprint, QuantizedFloat8, QuantizedPQ8, QuantizedSparse,
    FingerprintQuantizeError,
};
```

---

## Research Findings: Best Practices (2025-2026)

Based on comprehensive research of current quantization techniques:

### Industry Benchmarks for Quality Retention

| Method | Cosine Similarity Preservation | Source |
|--------|-------------------------------|--------|
| Float16 | ~99% | HuggingFace |
| INT8 Scalar | ~99% | HuggingFace |
| INT4 (calibrated) | ~98% | Elasticsearch 8.15 |
| PQ-8 | ~95% | Qdrant |
| Binary + Rescore | ~96% | Qdrant |

### Key Best Practices

1. **Outlier Handling**: Use 99th percentile quantiles instead of true min/max
   - Per-dimension calibration is more robust than global range
   - Prevents single outlier from degrading all values

2. **Scalar Quantization**: Maintains 99%+ accuracy across diverse embedding models
   - Works reliably with OpenAI, Cohere, Anthropic embeddings
   - 3.66x average speedup with INT8 vs float32

3. **PQ Codebook Training**: Minimum 10,000 samples for stable codebooks
   - Use Asymmetric Distance Computation (query=full precision, doc=quantized)
   - This task uses simplified mean-based PQ (no trained codebooks)

4. **SPLADE/Sparse Vectors**: Common quantization factor = 100-1000
   - Scale float importance weights to integers
   - Enables standard inverted index infrastructure

### Implementation Recommendation: Quantile-Based Scalar

For Float8 quantization, consider quantile-based clipping:

```rust
// More robust than simple min/max
fn calibrated_min_max(data: &[f32], quantile: f32) -> (f32, f32) {
    let mut sorted = data.to_vec();
    sorted.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let low_idx = ((1.0 - quantile) * sorted.len() as f32) as usize;
    let high_idx = (quantile * sorted.len() as f32) as usize;
    (sorted[low_idx], sorted[high_idx.min(sorted.len() - 1)])
}
```

### Sources

- HuggingFace: Binary and Scalar Embedding Quantization
- Qdrant: Quantization Documentation
- NVIDIA: FP8 for Deep Learning (E4M3 vs E5M2 formats)
- Elasticsearch 8.15: 4-bit quantization with dynamic quantile optimization

---

## Implementation Notes

### Float8 Quantization

```rust
fn quantize_float8_slice(data: &[f32], embedder: Embedder) -> Result<QuantizedFloat8, FingerprintQuantizeError> {
    // Validate no NaN/Infinity (AP-10)
    for (i, &v) in data.iter().enumerate() {
        if !v.is_finite() {
            return Err(FingerprintQuantizeError::InvalidValue { embedder, index: i });
        }
    }

    if data.is_empty() {
        return Ok(QuantizedFloat8 { data: vec![], min_val: 0.0, max_val: 0.0, original_dim: 0 });
    }

    let min_val = data.iter().cloned().fold(f32::INFINITY, f32::min);
    let max_val = data.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
    let range = max_val - min_val;

    let quantized = if range > f32::EPSILON {
        data.iter().map(|&v| ((v - min_val) / range * 255.0).round() as u8).collect()
    } else {
        vec![128u8; data.len()]  // Constant vector
    };

    Ok(QuantizedFloat8 { data: quantized, min_val, max_val, original_dim: data.len() })
}

fn dequantize_float8(q: &QuantizedFloat8) -> Vec<f32> {
    let range = q.max_val - q.min_val;
    if range <= f32::EPSILON {
        return vec![q.min_val; q.original_dim];
    }
    q.data.iter().map(|&v| (v as f32 / 255.0) * range + q.min_val).collect()
}
```

### PQ-8 Quantization (Simplified)

```rust
fn quantize_pq8(data: &[f32], num_subvectors: usize, embedder: Embedder) -> Result<QuantizedPQ8, FingerprintQuantizeError> {
    if data.is_empty() {
        return Ok(QuantizedPQ8 { codes: vec![], num_subvectors, original_dim: 0 });
    }

    if data.len() % num_subvectors != 0 {
        return Err(FingerprintQuantizeError::InvalidInput {
            embedder,
            message: format!("dim {} not divisible by {}", data.len(), num_subvectors),
        });
    }

    let sub_size = data.len() / num_subvectors;
    let codes: Vec<u8> = (0..num_subvectors).map(|i| {
        let sub = &data[i*sub_size..(i+1)*sub_size];
        let mean = sub.iter().sum::<f32>() / sub.len() as f32;
        ((mean.clamp(-1.0, 1.0) + 1.0) * 127.5) as u8
    }).collect();

    Ok(QuantizedPQ8 { codes, num_subvectors, original_dim: data.len() })
}

fn dequantize_pq8(q: &QuantizedPQ8) -> Vec<f32> {
    if q.original_dim == 0 { return vec![]; }
    let sub_size = q.original_dim / q.num_subvectors;
    let mut data = vec![0.0f32; q.original_dim];
    for (i, &code) in q.codes.iter().enumerate() {
        let mean = (code as f32 / 127.5) - 1.0;
        for j in (i*sub_size)..((i+1)*sub_size) {
            data[j] = mean;
        }
    }
    data
}
```

### Sparse Quantization

```rust
fn quantize_sparse(sv: &SparseVector) -> QuantizedSparse {
    if sv.is_empty() {
        return QuantizedSparse { indices: vec![], quantized_values: vec![], min_val: 0.0, max_val: 0.0 };
    }

    let min_val = sv.values.iter().cloned().fold(f32::INFINITY, f32::min);
    let max_val = sv.values.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
    let range = max_val - min_val;

    let qv = if range > f32::EPSILON {
        sv.values.iter().map(|&v| ((v - min_val) / range * 255.0).round() as u8).collect()
    } else {
        vec![128u8; sv.values.len()]
    };

    QuantizedSparse { indices: sv.indices.clone(), quantized_values: qv, min_val, max_val }
}

fn dequantize_sparse(qs: &QuantizedSparse) -> SparseVector {
    let range = qs.max_val - qs.min_val;
    let values: Vec<f32> = if range > f32::EPSILON {
        qs.quantized_values.iter().map(|&v| (v as f32 / 255.0) * range + qs.min_val).collect()
    } else {
        vec![qs.min_val; qs.quantized_values.len()]
    };
    SparseVector::new(qs.indices.clone(), values).unwrap_or_default()
}
```

---

## Validation Criteria

| Criterion | Threshold | Method |
|-----------|-----------|--------|
| Serialized size | < 15KB | `bincode::serialize(&qfp).unwrap().len()` |
| Float8 NRMSE | < 1% | `compute_nrmse()` |
| PQ8 NRMSE | < 10% | `compute_nrmse()` |
| Cosine similarity deviation | < 5% | Compare original vs dequantized pairs |
| All 13 embeddings | 100% success | Unit test each |
| Empty/zero vectors | No panic | Edge case tests |

---

## Full State Verification Protocol (MANDATORY)

### 1. Source of Truth

- **Size**: `bincode::serialize(&quantized).unwrap().len()`
- **Accuracy**: `dequantize_fingerprint(&quantized)` values
- **Similarity**: Cosine similarity before/after roundtrip

### 2. Execute & Inspect

After `quantize_fingerprint()`:

```rust
let qfp = quantize_fingerprint(&fp)?;
println!("VERIFY: Quantization OK");

let bytes = bincode::serialize(&qfp).unwrap();
println!("VERIFY: Size = {} bytes (target < 15000)", bytes.len());
assert!(bytes.len() < 15000);

let recovered = dequantize_fingerprint(&qfp)?;
println!("VERIFY: Dequantization OK");
println!("VERIFY: e1.len()={} (expect 1024)", recovered.e1_semantic.len());
```

### 3. Edge Cases (MUST TEST ALL 3)

| Case | Input | Expected | Verify |
|------|-------|----------|--------|
| Empty | `SemanticFingerprint::zeroed()` | Quantizes, size ~baseline | Print sizes |
| Max sparse | 1526 active in E6/E13 | Indices preserved | `qs.indices.len()` |
| Constant | All values = 0.5 | `min==max`, recovers 0.5 | Print first value |

### 4. Success Log (MUST PRODUCE)

```
=== FINGERPRINT QUANTIZATION VERIFICATION ===
Fingerprints tested: N
Size range: [X]B - [Y]B (threshold < 15000B)
Float8 NRMSE: max [Z]% (threshold < 1%)
PQ8 NRMSE: max [W]% (threshold < 10%)
Cosine deviation: max [V]% (threshold < 5%)
Edge cases: 3/3 PASSED
ALL VERIFICATIONS PASSED
```

---

## Test Commands

```bash
cargo test --package context-graph-core quantization::fingerprint -- --nocapture
cargo test --package context-graph-core quantization
cargo check --package context-graph-core
```

---

## Compression Summary

| Component | Raw | Quantized | Ratio |
|-----------|-----|-----------|-------|
| E1 (1024D PQ8, 32 subs) | 4KB | 32B | 128x |
| E2-E4 (3x512D Float8) | 6KB | 1.5KB | 4x |
| E5 (768D PQ8, 24 subs) | 3KB | 24B | 128x |
| E6 (~1500 sparse) | ~6KB | ~2.5KB | 2.4x |
| E7 (1536D PQ8, 48 subs) | 6KB | 48B | 128x |
| E8 (384D Float8) | 1.5KB | 384B | 4x |
| E9 (1024D Float8) | 4KB | 1KB | 4x |
| E10 (768D PQ8, 24 subs) | 3KB | 24B | 128x |
| E11 (384D Float8) | 1.5KB | 384B | 4x |
| E12 (~10 tok Float8) | ~5KB | ~1.3KB | 4x |
| E13 (~1500 sparse) | ~6KB | ~2.5KB | 2.4x |
| **TOTAL** | **~46KB** | **~10KB** | **~4.6x** |

---

## Execution Checklist

- [x] Read existing `src/quantization/` module
- [x] Read `src/types/fingerprint/semantic/fingerprint.rs`
- [x] Read `src/types/fingerprint/sparse.rs`
- [x] Create `src/quantization/fingerprint.rs`
- [x] Implement `FingerprintQuantizeError`
- [x] Implement `QuantizedFloat8` + helpers
- [x] Implement `QuantizedPQ8` + helpers
- [x] Implement `QuantizedSparse` + helpers
- [x] Implement `QuantizedSemanticFingerprint`
- [x] Implement `quantize_fingerprint()`
- [x] Implement `dequantize_fingerprint()`
- [x] Implement `estimated_size_bytes()`
- [x] Update `src/quantization/mod.rs`
- [x] Write unit tests per quantization type
- [x] Write edge case tests (empty, max sparse, constant)
- [x] Write cosine similarity test
- [x] Run Full State Verification Protocol
- [x] Print verification log with all thresholds
- [x] Confirm size < 15KB

---

## Completion Audit (2026-01-16)

### Implementation Summary

Created `crates/context-graph-core/src/quantization/fingerprint.rs` with:

- `FingerprintQuantizeError` - Error type with 4 variants (E_FP_QUANT_001-004)
- `QuantizedFloat8` - Min-max scalar quantization (4x compression)
- `QuantizedPQ8` - Simplified mean-based product quantization (128x compression)
- `QuantizedSparse` - Sparse vector quantization (indices preserved, values Float8)
- `QuantizedSemanticFingerprint` - Complete quantized fingerprint struct
- `quantize_fingerprint()` - Main quantization function
- `dequantize_fingerprint()` - Main dequantization function
- Helper functions: `validate_finite`, `check_dim`, `validate_fingerprint_dimensions`

### Full State Verification Protocol Results

```
=== FINGERPRINT QUANTIZATION VERIFICATION ===
Fingerprints tested: 3
Size range: [5302]B - [5302]B (threshold < 15000B) ✓
Float8 NRMSE: max 0.1131% (threshold < 1%) ✓
PQ-8 NRMSE: 0.9109% (threshold < 10%) ✓
Cosine deviation: max 0.0006% (threshold < 5%) ✓
Edge cases: 3/3 PASSED
  - Zeroed fingerprint: size=3752B ✓
  - Max sparse (1500 indices): indices preserved ✓
  - Constant vector (all 0.5): recovers ~0.5 ✓
ALL VERIFICATIONS PASSED
```

### Test Results

```
running 23 tests
test quantization::fingerprint::tests::test_dimension_mismatch_detection ... ok
test quantization::fingerprint::tests::test_error_codes ... ok
test quantization::fingerprint::tests::test_edge_case_constant_vector ... ok
test quantization::fingerprint::tests::test_edge_case_max_sparse ... ok
test quantization::fingerprint::tests::test_cosine_similarity_preservation ... ok
test quantization::fingerprint::tests::test_edge_case_zeroed_fingerprint ... ok
test quantization::fingerprint::tests::test_float8_empty_input ... ok
test quantization::fingerprint::tests::test_float8_quantize_dequantize_roundtrip ... ok
test quantization::fingerprint::tests::test_fingerprint_float8_accuracy ... ok
test quantization::fingerprint::tests::test_fingerprint_pq8_accuracy ... ok
test quantization::fingerprint::tests::test_fingerprint_quantize_dequantize_roundtrip ... ok
test quantization::fingerprint::tests::test_float8_constant_vector ... ok
test quantization::fingerprint::tests::test_fingerprint_size_under_threshold ... ok
test quantization::fingerprint::tests::test_float8_rejects_nan ... ok
test quantization::fingerprint::tests::test_float8_rejects_infinity ... ok
test quantization::fingerprint::tests::test_pq8_empty_input ... ok
test quantization::fingerprint::tests::test_pq8_compression_ratio ... ok
test quantization::fingerprint::tests::test_pq8_invalid_subvector_count ... ok
test quantization::fingerprint::tests::test_sparse_empty ... ok
test quantization::fingerprint::tests::test_pq8_quantize_dequantize_roundtrip ... ok
test quantization::fingerprint::tests::test_sparse_preserves_indices_exactly ... ok
test quantization::fingerprint::tests::test_sparse_quantize_dequantize_roundtrip ... ok
test quantization::fingerprint::tests::test_full_state_verification ... ok

test result: ok. 23 passed; 0 failed; 0 ignored; 0 measured
```

### Files Modified

1. **Created**: `crates/context-graph-core/src/quantization/fingerprint.rs` (1270 lines)
2. **Modified**: `crates/context-graph-core/src/quantization/mod.rs` (added exports)

### Compression Achieved

| Component | Target | Actual | Status |
|-----------|--------|--------|--------|
| Total size | < 15KB | 5.3KB | ✓ EXCEEDS |
| Float8 NRMSE | < 1% | 0.11% | ✓ |
| PQ-8 NRMSE | < 10% | 0.91% | ✓ |
| Cosine deviation | < 5% | 0.0006% | ✓ EXCEEDS |

### Code Simplifier Applied

Reduced from 1337 lines to 1270 lines (5% reduction) by:
- Extracting `validate_finite` helper function
- Extracting `check_dim` helper function
- Simplifying `validate_fingerprint_dimensions` from 70+ lines to 22 lines
- Removing redundant validation
- Streamlining E12 collection pattern

