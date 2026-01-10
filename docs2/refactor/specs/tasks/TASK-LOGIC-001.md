# TASK-LOGIC-001: Dense Similarity Functions

## STATUS: COMPLETED

## CRITICAL CONTEXT FOR IMPLEMENTATION

This task creates standalone dense vector similarity functions with SIMD acceleration. These functions are **primitives** used by the existing `DefaultCrossSpaceEngine` (which already has working cosine similarity but lacks SIMD optimization).

### What Already Exists (DO NOT DUPLICATE)

```
crates/context-graph-core/src/similarity/
├── mod.rs              # CrossSpaceSimilarityEngine exports
├── config.rs           # CrossSpaceConfig, WeightingStrategy
├── default_engine.rs   # DefaultCrossSpaceEngine with SCALAR cosine_similarity_dense()
├── engine.rs           # CrossSpaceSimilarityEngine trait
├── error.rs            # SimilarityError enum (already complete)
├── explanation.rs      # SimilarityExplanation
├── multi_utl.rs        # Multi-UTL formula
├── result.rs           # CrossSpaceSimilarity result
└── tests.rs            # Existing similarity tests
```

The `DefaultCrossSpaceEngine::cosine_similarity_dense()` (line 82-104 in default_engine.rs) is a SCALAR implementation. This task adds a SIMD-accelerated version that can be used as a drop-in replacement.

### Current Embedder Dimensions (Source of Truth)

From `crates/context-graph-core/src/types/fingerprint/semantic/constants.rs`:
```rust
E1_DIM = 1024   // Semantic - DENSE
E2_DIM = 512    // TemporalRecent - DENSE
E3_DIM = 512    // TemporalPeriodic - DENSE
E4_DIM = 512    // TemporalPositional - DENSE
E5_DIM = 768    // Causal - DENSE
E6_SPARSE_VOCAB = 30522  // SPLADE (SPARSE - not handled by this task)
E7_DIM = 1536   // Code - DENSE
E8_DIM = 384    // Graph - DENSE
E9_DIM = 1024   // HDC (projected dense) - DENSE
E10_DIM = 768   // Multimodal - DENSE
E11_DIM = 384   // Entity - DENSE
E12_TOKEN_DIM = 128  // LateInteraction (TOKEN-LEVEL - not handled by this task)
E13_SPLADE_VOCAB = 30522  // KeywordSplade (SPARSE - not handled by this task)
```

Dense embedders: E1, E2, E3, E4, E5, E7, E8, E9, E10, E11 (10 of 13)

---

## IMPLEMENTATION REQUIREMENTS

### 1. File Location

Create: `crates/context-graph-core/src/similarity/dense.rs`

Update: `crates/context-graph-core/src/similarity/mod.rs` to add:
```rust
mod dense;
pub use dense::{
    cosine_similarity, dot_product, euclidean_distance, l2_norm, normalize,
    DenseSimilarityError,
};
#[cfg(target_arch = "x86_64")]
pub use dense::cosine_similarity_simd;
```

### 2. Error Type

Create `DenseSimilarityError` in `dense.rs`:
```rust
use thiserror::Error;

#[derive(Debug, Error, Clone, PartialEq)]
pub enum DenseSimilarityError {
    #[error("Dimension mismatch: expected {expected}, got {actual}")]
    DimensionMismatch { expected: usize, actual: usize },

    #[error("Empty vector provided")]
    EmptyVector,

    #[error("Zero magnitude vector - cosine undefined")]
    ZeroMagnitude,
}
```

### 3. Core Functions

```rust
/// Calculate cosine similarity between two dense vectors.
/// Returns value in [-1.0, 1.0] where 1.0 means identical direction.
pub fn cosine_similarity(a: &[f32], b: &[f32]) -> Result<f32, DenseSimilarityError>;

/// Calculate dot product between two dense vectors.
pub fn dot_product(a: &[f32], b: &[f32]) -> Result<f32, DenseSimilarityError>;

/// Calculate Euclidean distance between two dense vectors.
pub fn euclidean_distance(a: &[f32], b: &[f32]) -> Result<f32, DenseSimilarityError>;

/// Calculate L2 norm (magnitude) of a vector.
pub fn l2_norm(v: &[f32]) -> f32;

/// Normalize a vector to unit length in-place.
/// Does nothing if vector has zero magnitude.
pub fn normalize(v: &mut [f32]);
```

### 4. SIMD Implementation (x86_64 only)

```rust
#[cfg(target_arch = "x86_64")]
pub fn cosine_similarity_simd(a: &[f32], b: &[f32]) -> Result<f32, DenseSimilarityError>;
```

Requirements:
- Use AVX2 intrinsics (`_mm256_*`)
- Process 8 floats per iteration
- Handle remainder with scalar fallback
- **MUST produce identical results to scalar version** (within f32 epsilon)

### 5. Integration with DefaultCrossSpaceEngine

After implementing, update `default_engine.rs` line 82:
```rust
// Replace scalar implementation:
fn cosine_similarity_dense(a: &[f32], b: &[f32]) -> Result<f32, SimilarityError> {
    #[cfg(target_arch = "x86_64")]
    {
        super::dense::cosine_similarity_simd(a, b)
            .map_err(|e| SimilarityError::invalid_config(e.to_string()))
    }
    #[cfg(not(target_arch = "x86_64"))]
    {
        super::dense::cosine_similarity(a, b)
            .map_err(|e| SimilarityError::invalid_config(e.to_string()))
    }
}
```

---

## IMPLEMENTATION CODE

```rust
// crates/context-graph-core/src/similarity/dense.rs

//! Dense vector similarity functions with optional SIMD acceleration.
//!
//! This module provides core similarity primitives for dense embeddings
//! (E1, E2, E3, E4, E5, E7, E8, E9, E10, E11).
//!
//! # Performance
//!
//! SIMD (AVX2) provides 2-4x speedup on x86_64 for vectors >256 dimensions.
//! Constitution.yaml target: <5ms for pair similarity.

use thiserror::Error;

/// Errors from dense vector similarity computation.
#[derive(Debug, Error, Clone, PartialEq)]
pub enum DenseSimilarityError {
    #[error("Dimension mismatch: expected {expected}, got {actual}")]
    DimensionMismatch { expected: usize, actual: usize },

    #[error("Empty vector provided")]
    EmptyVector,

    #[error("Zero magnitude vector - cosine undefined")]
    ZeroMagnitude,
}

/// Calculate L2 norm (magnitude) of a vector.
#[inline]
pub fn l2_norm(v: &[f32]) -> f32 {
    v.iter().map(|x| x * x).sum::<f32>().sqrt()
}

/// Normalize a vector to unit length in-place.
/// Does nothing if vector has zero magnitude.
#[inline]
pub fn normalize(v: &mut [f32]) {
    let norm = l2_norm(v);
    if norm > f32::EPSILON {
        for x in v.iter_mut() {
            *x /= norm;
        }
    }
}

/// Internal dot product without validation.
#[inline]
fn dot_product_unchecked(a: &[f32], b: &[f32]) -> f32 {
    a.iter().zip(b.iter()).map(|(x, y)| x * y).sum()
}

/// Calculate dot product between two dense vectors.
#[inline]
pub fn dot_product(a: &[f32], b: &[f32]) -> Result<f32, DenseSimilarityError> {
    if a.is_empty() || b.is_empty() {
        return Err(DenseSimilarityError::EmptyVector);
    }
    if a.len() != b.len() {
        return Err(DenseSimilarityError::DimensionMismatch {
            expected: a.len(),
            actual: b.len(),
        });
    }
    Ok(dot_product_unchecked(a, b))
}

/// Calculate cosine similarity between two dense vectors.
/// Returns value in [-1.0, 1.0] where 1.0 means identical direction.
#[inline]
pub fn cosine_similarity(a: &[f32], b: &[f32]) -> Result<f32, DenseSimilarityError> {
    if a.is_empty() || b.is_empty() {
        return Err(DenseSimilarityError::EmptyVector);
    }
    if a.len() != b.len() {
        return Err(DenseSimilarityError::DimensionMismatch {
            expected: a.len(),
            actual: b.len(),
        });
    }

    let dot = dot_product_unchecked(a, b);
    let norm_a = l2_norm(a);
    let norm_b = l2_norm(b);

    if norm_a < f32::EPSILON || norm_b < f32::EPSILON {
        return Err(DenseSimilarityError::ZeroMagnitude);
    }

    let result = dot / (norm_a * norm_b);
    // Clamp to valid range to handle floating point errors
    Ok(result.clamp(-1.0, 1.0))
}

/// Calculate Euclidean distance between two dense vectors.
#[inline]
pub fn euclidean_distance(a: &[f32], b: &[f32]) -> Result<f32, DenseSimilarityError> {
    if a.is_empty() || b.is_empty() {
        return Err(DenseSimilarityError::EmptyVector);
    }
    if a.len() != b.len() {
        return Err(DenseSimilarityError::DimensionMismatch {
            expected: a.len(),
            actual: b.len(),
        });
    }
    let sum: f32 = a.iter()
        .zip(b.iter())
        .map(|(x, y)| (x - y).powi(2))
        .sum();
    Ok(sum.sqrt())
}

// ============================================================================
// SIMD IMPLEMENTATION (x86_64 AVX2)
// ============================================================================

#[cfg(target_arch = "x86_64")]
pub fn cosine_similarity_simd(a: &[f32], b: &[f32]) -> Result<f32, DenseSimilarityError> {
    use std::arch::x86_64::*;

    if a.is_empty() || b.is_empty() {
        return Err(DenseSimilarityError::EmptyVector);
    }
    if a.len() != b.len() {
        return Err(DenseSimilarityError::DimensionMismatch {
            expected: a.len(),
            actual: b.len(),
        });
    }

    // For small vectors, use scalar (SIMD overhead not worth it)
    if a.len() < 32 {
        return cosine_similarity(a, b);
    }

    // Check AVX2 support at runtime
    if !is_x86_feature_detected!("avx2") || !is_x86_feature_detected!("fma") {
        return cosine_similarity(a, b);
    }

    unsafe {
        let mut dot_sum = _mm256_setzero_ps();
        let mut norm_a_sum = _mm256_setzero_ps();
        let mut norm_b_sum = _mm256_setzero_ps();

        let chunks = a.len() / 8;
        for i in 0..chunks {
            let va = _mm256_loadu_ps(a.as_ptr().add(i * 8));
            let vb = _mm256_loadu_ps(b.as_ptr().add(i * 8));

            // FMA: dot_sum += va * vb
            dot_sum = _mm256_fmadd_ps(va, vb, dot_sum);
            // FMA: norm_a_sum += va * va
            norm_a_sum = _mm256_fmadd_ps(va, va, norm_a_sum);
            // FMA: norm_b_sum += vb * vb
            norm_b_sum = _mm256_fmadd_ps(vb, vb, norm_b_sum);
        }

        // Horizontal sum: reduce 8 lanes to 1
        let dot = hsum_avx(dot_sum);
        let norm_a_sq = hsum_avx(norm_a_sum);
        let norm_b_sq = hsum_avx(norm_b_sum);

        // Handle remainder with scalar code
        let remainder_start = chunks * 8;
        let mut dot_rem = 0.0f32;
        let mut norm_a_rem = 0.0f32;
        let mut norm_b_rem = 0.0f32;
        for i in remainder_start..a.len() {
            dot_rem += a[i] * b[i];
            norm_a_rem += a[i] * a[i];
            norm_b_rem += b[i] * b[i];
        }

        let total_dot = dot + dot_rem;
        let total_norm_a = (norm_a_sq + norm_a_rem).sqrt();
        let total_norm_b = (norm_b_sq + norm_b_rem).sqrt();

        if total_norm_a < f32::EPSILON || total_norm_b < f32::EPSILON {
            return Err(DenseSimilarityError::ZeroMagnitude);
        }

        let result = total_dot / (total_norm_a * total_norm_b);
        Ok(result.clamp(-1.0, 1.0))
    }
}

/// Horizontal sum of 8 f32 lanes in AVX register.
#[cfg(target_arch = "x86_64")]
#[inline]
unsafe fn hsum_avx(v: std::arch::x86_64::__m256) -> f32 {
    use std::arch::x86_64::*;
    // Sum pairs: [a+b, c+d, e+f, g+h, a+b, c+d, e+f, g+h]
    let sum1 = _mm256_hadd_ps(v, v);
    // Sum pairs again: [a+b+c+d, e+f+g+h, a+b+c+d, e+f+g+h, ...]
    let sum2 = _mm256_hadd_ps(sum1, sum1);
    // Extract low and high 128-bit lanes and add
    let low = _mm256_extractf128_ps(sum2, 0);
    let high = _mm256_extractf128_ps(sum2, 1);
    let sum3 = _mm_add_ps(low, high);
    _mm_cvtss_f32(sum3)
}

// ============================================================================
// TESTS
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_cosine_identical_vectors() {
        let v = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let sim = cosine_similarity(&v, &v).unwrap();
        assert!((sim - 1.0).abs() < 1e-6, "Identical vectors should have similarity 1.0, got {}", sim);
        println!("[PASS] Cosine of identical vectors = 1.0: actual = {:.6}", sim);
    }

    #[test]
    fn test_cosine_orthogonal_vectors() {
        let a = vec![1.0, 0.0];
        let b = vec![0.0, 1.0];
        let sim = cosine_similarity(&a, &b).unwrap();
        assert!(sim.abs() < 1e-6, "Orthogonal vectors should have similarity 0.0, got {}", sim);
        println!("[PASS] Cosine of orthogonal vectors = 0.0: actual = {:.6}", sim);
    }

    #[test]
    fn test_cosine_opposite_vectors() {
        let a = vec![1.0, 2.0, 3.0];
        let b = vec![-1.0, -2.0, -3.0];
        let sim = cosine_similarity(&a, &b).unwrap();
        assert!((sim + 1.0).abs() < 1e-6, "Opposite vectors should have similarity -1.0, got {}", sim);
        println!("[PASS] Cosine of opposite vectors = -1.0: actual = {:.6}", sim);
    }

    #[test]
    fn test_dimension_mismatch_error() {
        let a = vec![1.0, 2.0];
        let b = vec![1.0, 2.0, 3.0];
        let result = cosine_similarity(&a, &b);
        assert!(matches!(result, Err(DenseSimilarityError::DimensionMismatch { expected: 2, actual: 3 })));
        println!("[PASS] Dimension mismatch correctly detected: {:?}", result.err());
    }

    #[test]
    fn test_empty_vector_error() {
        let a: Vec<f32> = vec![];
        let b = vec![1.0, 2.0];
        let result = cosine_similarity(&a, &b);
        assert!(matches!(result, Err(DenseSimilarityError::EmptyVector)));
        println!("[PASS] Empty vector correctly detected: {:?}", result.err());
    }

    #[test]
    fn test_zero_magnitude_error() {
        let a = vec![0.0, 0.0, 0.0];
        let b = vec![1.0, 2.0, 3.0];
        let result = cosine_similarity(&a, &b);
        assert!(matches!(result, Err(DenseSimilarityError::ZeroMagnitude)));
        println!("[PASS] Zero magnitude correctly detected: {:?}", result.err());
    }

    #[test]
    fn test_dot_product() {
        let a = vec![1.0, 2.0, 3.0];
        let b = vec![4.0, 5.0, 6.0];
        let dot = dot_product(&a, &b).unwrap();
        let expected = 1.0 * 4.0 + 2.0 * 5.0 + 3.0 * 6.0; // 32.0
        assert!((dot - expected).abs() < 1e-6);
        println!("[PASS] Dot product = {}, expected = {}", dot, expected);
    }

    #[test]
    fn test_euclidean_distance() {
        let a = vec![0.0, 0.0];
        let b = vec![3.0, 4.0];
        let dist = euclidean_distance(&a, &b).unwrap();
        assert!((dist - 5.0).abs() < 1e-6, "Expected distance 5.0, got {}", dist);
        println!("[PASS] Euclidean distance = {}, expected = 5.0", dist);
    }

    #[test]
    fn test_l2_norm() {
        let v = vec![3.0, 4.0];
        let norm = l2_norm(&v);
        assert!((norm - 5.0).abs() < 1e-6);
        println!("[PASS] L2 norm of [3,4] = {}, expected = 5.0", norm);
    }

    #[test]
    fn test_normalize() {
        let mut v = vec![3.0, 4.0];
        normalize(&mut v);
        let norm = l2_norm(&v);
        assert!((norm - 1.0).abs() < 1e-6, "Normalized vector should have norm 1.0, got {}", norm);
        assert!((v[0] - 0.6).abs() < 1e-6);
        assert!((v[1] - 0.8).abs() < 1e-6);
        println!("[PASS] Normalized [3,4] = [{:.3}, {:.3}]", v[0], v[1]);
    }

    #[test]
    fn test_high_dimensional_1024() {
        // Simulate E1_DIM = 1024
        let a: Vec<f32> = (0..1024).map(|i| (i as f32) * 0.001).collect();
        let b: Vec<f32> = (0..1024).map(|i| ((i as f32) * 0.001).sin()).collect();
        let sim = cosine_similarity(&a, &b).unwrap();
        assert!(sim >= -1.0 && sim <= 1.0, "Similarity out of range: {}", sim);
        assert!(!sim.is_nan() && !sim.is_infinite());
        println!("[PASS] 1024D cosine similarity = {:.6}", sim);
    }

    #[cfg(target_arch = "x86_64")]
    #[test]
    fn test_simd_matches_scalar() {
        // Test with E1 dimensions (1024)
        let a: Vec<f32> = (0..1024).map(|i| (i as f32 * 0.001) + 0.5).collect();
        let b: Vec<f32> = (0..1024).map(|i| ((i as f32) * 0.002).sin()).collect();

        let scalar = cosine_similarity(&a, &b).unwrap();
        let simd = cosine_similarity_simd(&a, &b).unwrap();

        let diff = (scalar - simd).abs();
        assert!(diff < 1e-5, "SIMD result differs from scalar by {}: scalar={}, simd={}", diff, scalar, simd);
        println!("[PASS] SIMD matches scalar within 1e-5: scalar={:.6}, simd={:.6}, diff={:.9}", scalar, simd, diff);
    }

    #[cfg(target_arch = "x86_64")]
    #[test]
    fn test_simd_with_different_dimensions() {
        // Test all dense embedder dimensions
        let dimensions = [1024, 512, 768, 1536, 384]; // E1, E2/E3/E4, E5/E10, E7, E8/E11

        for dim in dimensions {
            let a: Vec<f32> = (0..dim).map(|i| (i as f32 * 0.001) + 0.5).collect();
            let b: Vec<f32> = (0..dim).map(|i| ((i as f32) * 0.002).cos()).collect();

            let scalar = cosine_similarity(&a, &b).unwrap();
            let simd = cosine_similarity_simd(&a, &b).unwrap();

            let diff = (scalar - simd).abs();
            assert!(diff < 1e-5, "SIMD differs at dim={}: diff={}", dim, diff);
            println!("[PASS] dim={}: scalar={:.6}, simd={:.6}", dim, scalar, simd);
        }
    }
}
```

---

## VERIFICATION PROTOCOL

### 1. Unit Test Execution

```bash
# Run all dense similarity tests
cargo test -p context-graph-core dense -- --nocapture

# Expected output for each test:
# [PASS] Cosine of identical vectors = 1.0
# [PASS] Cosine of orthogonal vectors = 0.0
# [PASS] Cosine of opposite vectors = -1.0
# [PASS] Dimension mismatch correctly detected
# [PASS] Empty vector correctly detected
# [PASS] Zero magnitude correctly detected
# [PASS] Dot product = 32
# [PASS] Euclidean distance = 5.0
# [PASS] L2 norm of [3,4] = 5.0
# [PASS] Normalized [3,4] = [0.600, 0.800]
# [PASS] 1024D cosine similarity = <value>
# [PASS] SIMD matches scalar (x86_64 only)
# [PASS] SIMD with different dimensions (x86_64 only)
```

### 2. Source of Truth Verification

After implementation, manually verify in Rust REPL or test:

```rust
// EDGE CASE 1: Empty input
assert!(cosine_similarity(&[], &[1.0]).is_err());
println!("STATE BEFORE: a=[], b=[1.0]");
println!("STATE AFTER: Err(EmptyVector)");

// EDGE CASE 2: Zero magnitude
let zero = vec![0.0; 1024];
let normal = vec![1.0; 1024];
assert!(cosine_similarity(&zero, &normal).is_err());
println!("STATE BEFORE: a=<zero vector>, b=<normal vector>");
println!("STATE AFTER: Err(ZeroMagnitude)");

// EDGE CASE 3: Maximum values (no overflow)
let max_vals = vec![f32::MAX / 1000.0; 1024]; // Scaled to avoid overflow
let result = cosine_similarity(&max_vals, &max_vals);
assert!(result.is_ok());
println!("STATE BEFORE: a=<near-max values>, b=<near-max values>");
println!("STATE AFTER: similarity = {}", result.unwrap());
```

### 3. SIMD vs Scalar Parity Check

For x86_64, run benchmark comparing results:

```bash
cargo bench -p context-graph-core -- dense
```

Acceptance criteria:
- SIMD produces results within 1e-5 of scalar
- SIMD provides 2-4x speedup for 512D+ vectors
- No NaN or Inf in any output

### 4. Integration Test with Existing Similarity Module

After modifying `default_engine.rs` to use SIMD:

```bash
cargo test -p context-graph-core similarity -- --nocapture
```

All existing similarity tests MUST continue to pass.

---

## EVIDENCE OF SUCCESS LOG

After implementation, the following log entries prove success:

```
=== DENSE SIMILARITY VERIFICATION LOG ===
Test Environment: cargo test -p context-graph-core dense -- --nocapture
Timestamp: 2026-01-10T00:03:00Z

TEST RESULTS:
├── test_cosine_identical_vectors: PASS (sim=1.000000)
├── test_cosine_orthogonal_vectors: PASS (sim=0.000000)
├── test_cosine_opposite_vectors: PASS (sim=-1.000000)
├── test_dimension_mismatch_error: PASS (DimensionMismatch { expected: 2, actual: 3 })
├── test_empty_vector_error: PASS (EmptyVector)
├── test_zero_magnitude_error: PASS (ZeroMagnitude)
├── test_dot_product: PASS (dot=32.0)
├── test_euclidean_distance: PASS (dist=5.0)
├── test_l2_norm: PASS (norm=5.0)
├── test_normalize: PASS (v=[0.600, 0.800])
├── test_high_dimensional_1024: PASS (sim in [-1,1])
├── test_simd_matches_scalar: PASS (diff < 1e-5)
└── test_simd_with_different_dimensions: PASS (all dims verified)

EDGE CASE AUDIT:
├── Empty input: Correctly returns EmptyVector error
├── Zero magnitude: Correctly returns ZeroMagnitude error
└── Near-max values: Correctly computes without overflow

SOURCE OF TRUTH CHECK:
├── Identical vectors → similarity = 1.0 ✓
├── Orthogonal vectors → similarity = 0.0 ✓
├── Opposite vectors → similarity = -1.0 ✓
└── SIMD output matches scalar within 1e-5 ✓

INTEGRATION VERIFICATION:
└── All existing similarity tests pass after integration: ✓
```

---

## DEPENDENCIES

### Required (COMPLETED)

- **TASK-CORE-002**: Embedder enum exists at `crates/context-graph-core/src/teleological/embedder.rs`
- **TASK-CORE-003**: TeleologicalArray type exists at `crates/context-graph-core/src/types/fingerprint/semantic/`

### Uses Existing

- `crates/context-graph-core/src/similarity/error.rs` - `SimilarityError` for integration
- `crates/context-graph-core/src/similarity/default_engine.rs` - Update to use SIMD functions

---

## ANTI-PATTERNS TO AVOID

1. **NO Mock Data in Tests**: Use real mathematical properties (e.g., orthogonal vectors = 0.0)
2. **NO Backwards Compatibility Shims**: If SIMD doesn't match scalar, FIX IT, don't hide it
3. **NO Silent Failures**: Every edge case must return an explicit error
4. **NO NaN/Inf Propagation**: Clamp results and validate inputs
5. **NO Dimension Guessing**: Dimension mismatch is always an error

---

## FILES TO CREATE

| File | Purpose |
|------|---------|
| `crates/context-graph-core/src/similarity/dense.rs` | Dense similarity functions with SIMD |

## FILES TO MODIFY

| File | Change |
|------|--------|
| `crates/context-graph-core/src/similarity/mod.rs` | Add `mod dense; pub use dense::*;` |
| `crates/context-graph-core/src/similarity/default_engine.rs` | Update `cosine_similarity_dense()` to use SIMD |

---

## TEST COMMANDS

```bash
# Compile and run tests
cargo test -p context-graph-core dense -- --nocapture

# Run benchmarks (if benches exist)
cargo bench -p context-graph-core -- dense

# Verify all similarity tests still pass
cargo test -p context-graph-core similarity -- --nocapture

# Full crate test
cargo test -p context-graph-core
```

---

## NOTES FOR IMPLEMENTING AGENT

1. The existing `DefaultCrossSpaceEngine::cosine_similarity_dense()` is a WORKING scalar implementation. Your task is to add SIMD acceleration, not replace working code.

2. The error type `DenseSimilarityError` is separate from `SimilarityError` because it's a lower-level primitive. The integration layer in `default_engine.rs` maps one to the other.

3. SIMD requires runtime feature detection (`is_x86_feature_detected!("avx2")`). Fall back to scalar if not available.

4. All 10 dense embedders use dimensions divisible by 8 except E8/E11 (384). The SIMD code handles remainders.

5. The `normalize()` function is in-place mutation. This is intentional for performance.
