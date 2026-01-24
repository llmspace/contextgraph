# GPU Usage Analysis Report

**Generated:** 2026-01-24
**Codebase:** Context Graph v6.3.0
**Target Hardware:** RTX 5090 (32GB GDDR7, Compute 12.0, CUDA 13.1)

---

## Executive Summary

The Context Graph codebase demonstrates **excellent GPU adoption** for neural network inference, with all 11 pretrained embedding models properly using CUDA. However, **significant CPU-bound operations** exist in the retrieval pipeline, particularly in similarity computation, that could benefit from GPU acceleration.

| Category | Status | Impact |
|----------|--------|--------|
| Model Inference (E1-E13) | GPU | N/A - Already optimized |
| FAISS Index Operations | GPU | N/A - Already optimized |
| Similarity Computation | CPU | **HIGH** - Called 130K+ times/query |
| Quantization Batch Ops | CPU (rayon) | MEDIUM - Could use GPU |
| Tokenization | CPU | LOW - I/O bound, CPU is correct |
| Result Aggregation | CPU | LOW - Small data, CPU is correct |

---

## Critical Findings: CPU Operations That Should Use GPU

### 1. Similarity Computation in Storage Layer (HIGH PRIORITY)

**Files:**
- `crates/context-graph-storage/src/teleological/rocksdb_store/helpers.rs:6`
- `crates/context-graph-storage/src/teleological/indexes/metrics.rs:48-95`
- `crates/context-graph-storage/src/teleological/indexes/metrics.rs:137-165`

**Current Implementation:**
```rust
// helpers.rs:6-27 - Pure CPU scalar loop
pub fn compute_cosine_similarity(a: &[f32], b: &[f32]) -> f32 {
    let mut dot = 0.0f32;
    let mut norm_a = 0.0f32;
    let mut norm_b = 0.0f32;
    for i in 0..a.len() {
        dot += a[i] * b[i];
        norm_a += a[i] * a[i];
        norm_b += b[i] * b[i];
    }
    dot / (norm_a.sqrt() * norm_b.sqrt())
}
```

**Impact:** This function is called in `index_ops.rs` lines 161-259 for EVERY candidate document across ALL 13 embedders. For a typical query:
- 10,000 candidate documents
- 13 embedder comparisons each
- = **130,000 cosine similarity computations PER QUERY**

**Recommendation:** Batch similarity computation on GPU:
```rust
// GPU batched version (conceptual)
pub fn batch_cosine_similarity_gpu(
    queries: &Tensor,      // [num_queries, dim]
    candidates: &Tensor,   // [num_candidates, dim]
) -> Tensor {             // [num_queries, num_candidates]
    // Single GPU matmul: O(1) kernel launch
    let dots = queries.matmul(&candidates.transpose(0, 1))?;
    let norms_q = queries.sqr()?.sum(1)?.sqrt()?;
    let norms_c = candidates.sqr()?.sum(1)?.sqrt()?;
    dots / norms_q.outer(&norms_c)?
}
```

---

### 2. MaxSim Token-Level Similarity (MEDIUM PRIORITY)

**File:** `crates/context-graph-storage/src/teleological/search/maxsim.rs:70-165`

**Current Implementation:**
```rust
// Line 70: AVX2-optimized but still CPU
pub fn cosine_similarity_128d(a: &[f32], b: &[f32]) -> f32 {
    #[cfg(target_feature = "avx2")]
    unsafe { cosine_similarity_avx2(a, b) }  // SIMD but CPU

    #[cfg(not(target_feature = "avx2"))]
    cosine_similarity_scalar(a, b)  // Scalar CPU fallback
}
```

**Called from:** `compute_maxsim_direct()` line 213 and `compute_maxsim_batched()` line 324

**Impact:** E12 ColBERT reranking uses MaxSim which computes token-level similarities:
- 128D tokens per document
- O(query_tokens × doc_tokens × candidates) comparisons
- Currently uses rayon for parallel CPU execution (line 257)

**Recommendation:** GPU kernel for batched MaxSim:
- Load all document token matrices to GPU once
- Compute all query-document MaxSim scores in parallel
- Expected speedup: 10-50x for large candidate sets

---

### 3. Quantization Batch Operations (MEDIUM PRIORITY)

**File:** `crates/context-graph-core/src/quantization/batch.rs:7-41`

**Current Implementation:**
```rust
use rayon::prelude::*;

pub fn batch_quantize<T: Quantizable>(vectors: &[Vec<f32>]) -> Vec<T> {
    vectors.par_iter().map(|v| T::quantize(v)).collect()  // CPU parallel
}

pub fn batch_dequantize<T: Quantizable>(quantized: &[T]) -> Vec<Vec<f32>> {
    quantized.par_iter().map(T::dequantize).collect()  // CPU parallel
}
```

**Impact:** Quantization/dequantization of embeddings for storage.
- Current: rayon CPU parallelism
- Could use: GPU parallel quantization kernels

**Recommendation:** For Product Quantization (PQ-8) codebook operations:
- GPU codebook distance computation
- Batched assignment to centroids

---

### 4. Multi-Space Search Aggregation (LOW PRIORITY)

**File:** `crates/context-graph-storage/src/teleological/search/multi/executor.rs:149-229`

**Current Implementation:**
```rust
// Line 149-151: rayon CPU parallel per-embedder search
spaces.into_par_iter().map(|space| search_single_space(space, query))
```

**Impact:** Aggregates results from individual embedder searches.
- Individual searches already use GPU (via FAISS)
- Aggregation is small data (top-K results per space)
- CPU is appropriate for this phase

**Status:** No change recommended - CPU aggregation is correct here.

---

## Areas Where CPU Usage is Correct (No Change Needed)

### 1. Tokenization (CPU - Correct)

**Files:** All model `forward.rs` files using `tokenizers::Tokenizer`

**Rationale:**
- Tokenization is I/O and string processing bound
- HuggingFace tokenizers are highly optimized for CPU
- GPU tokenization (e.g., cudf) adds complexity for minimal gain
- Sequence lengths are typically short (< 512 tokens)

### 2. Custom Temporal Models E2, E3, E4 (CPU - Correct)

**Files:**
- `crates/context-graph-embeddings/src/models/custom/temporal_recent/model.rs`
- `crates/context-graph-embeddings/src/models/custom/temporal_periodic/model.rs`
- `crates/context-graph-embeddings/src/models/custom/temporal_positional/`

**Rationale:**
- Simple mathematical functions (exp, sin, cos)
- No matrix operations
- Output dimensions are small (512D)
- GPU kernel launch overhead would exceed compute time

### 3. HDC Embedder E9 (CPU - Correct)

**File:** `crates/context-graph-embeddings/src/models/custom/hdc/model.rs`

**Rationale:**
- Hypervector operations are bitwise (XOR, AND)
- Designed for neuromorphic/edge computing
- CPU bit operations are extremely fast
- 10,000D but binary operations, not FP32

### 4. Result Ranking and Sorting (CPU - Correct)

**Files:** Various RRF and result aggregation in search pipeline

**Rationale:**
- Operating on small result sets (top-100 to top-1000)
- Comparison-based sorting is branchy (poor GPU utilization)
- CPU is more efficient for small-N sorts

---

## Test Code Using Device::Cpu (MINOR)

**File:** `crates/context-graph-embeddings/src/models/pretrained/contextual/model.rs`

**Lines:**
- Line 1029: `let device = Device::Cpu;` (test_layer_norm_shape_compatibility)
- Line 1053: `let device = Device::Cpu;` (test_layer_norm_different_batch_sizes)
- Line 1076: `let device = Device::Cpu;` (test_layer_norm_normalizes_output)
- Line 1103: `let device = Device::Cpu;` (test_layer_norm_different_sequence_lengths)

**Status:** Test code only. No production impact. Tests can run on CPU.

---

## GPU Adoption Summary by Module

### Pretrained Models (All GPU)

| Model | File | GPU Status | Line |
|-------|------|------------|------|
| E1 Semantic | `semantic/loader.rs` | `init_gpu()` | 41 |
| E5 Causal | `causal/forward.rs` | `init_gpu()` | GPU |
| E6 Sparse (SPLADE) | `sparse/loader.rs` | GPU loading | GPU |
| E7 Code | `code/forward.rs` | GPU inference | GPU |
| E8 Graph | `graph/model.rs` | `init_gpu()` | 97, 138 |
| E9 Multimodal | `multimodal/model.rs` | `init_gpu()` | 93 |
| E10 Contextual | `contextual/model.rs` | `init_gpu()` | 112 |
| E11 Entity (old) | `entity/model.rs` | `init_gpu()` | 64 |
| E11 KEPLER (new) | `kepler/model.rs` | `init_gpu()` | 64 |
| E12 LateInteraction | `late_interaction/model.rs` | `init_gpu()` | 120 |
| E13 SPLADE | (shared with E6) | GPU | GPU |

### CUDA/FAISS Integration (GPU)

| Component | File | Status |
|-----------|------|--------|
| FAISS FFI | `context-graph-cuda/src/ffi/faiss.rs` | GPU index support |
| GPU Resources | `faiss_StandardGpuResources_new()` | GPU memory mgmt |
| CPU→GPU Transfer | `faiss_index_cpu_to_gpu()` | Index migration |
| CUDA Allocator | `warm/cuda_alloc/` | Non-evictable VRAM |
| Inference Engine | `warm/inference/engine.rs:81` | `Device::new_cuda()` |

### GPU Configuration (Properly Configured)

**File:** `crates/context-graph-embeddings/src/config/gpu.rs`

| Setting | Value | Purpose |
|---------|-------|---------|
| `enabled` | `true` (default) | GPU acceleration on |
| `device_ids` | `[0]` | Primary GPU |
| `memory_fraction` | `0.9` | Use 90% of VRAM (28.8GB) |
| `use_cuda_graphs` | `true` | Kernel fusion enabled |
| `use_mixed_precision` | `true` | FP16/BF16 enabled |

---

## Recommendations Priority

### P0 - Critical (Implement First)

1. **GPU-Batched Cosine Similarity in Storage Layer**
   - Files: `helpers.rs`, `metrics.rs`
   - Expected speedup: 50-100x for query path
   - Implementation: Batch all candidates, single matmul kernel

### P1 - High (Implement Soon)

2. **GPU MaxSim Kernel for E12 Reranking**
   - File: `maxsim.rs`
   - Expected speedup: 10-50x for reranking phase
   - Implementation: Custom CUDA kernel or cuBLAS batched GEMM

### P2 - Medium (Consider)

3. **GPU Quantization for Storage**
   - File: `batch.rs`
   - Expected speedup: 5-20x for bulk embedding storage
   - Implementation: GPU PQ codebook operations

### P3 - Low (Optional)

4. **Test GPU Consistency**
   - Files: Test files with `Device::Cpu`
   - Low priority - tests work correctly

---

## Architecture Notes

### Why Some CPU Usage is Correct

1. **Amdahl's Law**: GPU overhead (kernel launch, memory transfer) exceeds benefit for small operations
2. **Memory Bandwidth**: Some operations are memory-bound, not compute-bound
3. **Branching**: Sorting/ranking has many branches - poor GPU utilization
4. **I/O Bound**: Tokenization is string processing, not matrix math

### GPU Transfer Costs

For GPU to be beneficial, compute must exceed:
- ~10μs kernel launch overhead
- ~1GB/s PCIe transfer for data movement
- ~5μs memory allocation

Similarity of 1000 vectors × 768D:
- CPU: ~1ms (AVX2 SIMD)
- GPU: ~0.05ms compute + ~0.5ms transfer = ~0.55ms
- **GPU wins at scale (>1000 vectors)**

---

## Conclusion

The Context Graph codebase has **excellent GPU adoption** for neural network inference. The primary opportunity for improvement is **batched similarity computation in the retrieval pipeline**, which is currently CPU-bound and called O(candidates × embedders) times per query.

Implementing GPU-batched cosine similarity in the storage layer would yield the highest performance improvement with minimal architectural changes.
