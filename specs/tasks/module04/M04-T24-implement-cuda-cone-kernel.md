---
id: "M04-T24"
title: "Implement Cone Membership CUDA Kernel"
description: |
  Implement cone_check_batch CUDA kernel for batch entailment cone membership.
  Input: cones[n_cones][65] (64 apex coords + 1 aperture), points[n_pts][64]
  Output: scores[n_cones][n_pts]
  Performance: <2ms for 1K x 1K membership matrix.
layer: "surface"
status: "done"
priority: "high"
estimated_hours: 6
sequence: 32
depends_on:
  - "M04-T07"  # EntailmentCone containment logic (COMPLETED)
  - "M04-T23"  # Poincare Distance CUDA Kernel (provides pattern to follow)
spec_refs:
  - "TECH-GRAPH-004 Section 10.2"
  - "constitution.yaml perf.latency.entailment_check: <1ms"
files_to_create:
  - path: "crates/context-graph-cuda/kernels/cone_check.cu"
    description: "CUDA kernel for batch cone membership checking"
  - path: "crates/context-graph-cuda/src/cone.rs"
    description: "Rust FFI wrapper for cone CUDA kernel"
  - path: "crates/context-graph-cuda/tests/cuda_cone_test.rs"
    description: "Integration tests for cone CUDA operations"
files_to_modify:
  - path: "crates/context-graph-cuda/src/lib.rs"
    description: "Add cone module export"
  - path: "crates/context-graph-cuda/build.rs"
    description: "Add cone_check.cu kernel compilation"
  - path: "crates/context-graph-cuda/Cargo.toml"
    description: "Add cc build dependency"
---

## CRITICAL: Read This First

**YOU ARE AN AI AGENT WITH A FRESH CONTEXT WINDOW.** This task document is your single source of truth. Do not make assumptions. Follow these rules:

1. **NO BACKWARDS COMPATIBILITY** - System must work after changes or fail fast with robust error logging
2. **NO WORKAROUNDS/FALLBACKS** - If something doesn't work, error out immediately
3. **NO MOCK DATA IN TESTS** - Use real data and verify actual outcomes
4. **VERIFY OUTPUTS PHYSICALLY** - Check databases, files, GPU memory to prove work was done
5. **USE SHERLOCK-HOLMES AGENT** at the end to forensically verify completion

---

## Current Project State (Audited 2026-01-04)

### Existing CUDA Crate Structure
```
crates/context-graph-cuda/
├── Cargo.toml           # Features: default=["stub"], cuda, cudnn
├── build.rs             # Exists, currently minimal (needs update)
├── kernels/
│   └── poincare_distance.cu   # Reference implementation from M04-T23
├── src/
│   ├── lib.rs           # Exports: error, poincare, ops, stub modules
│   ├── error.rs         # CudaError enum with 5 variants
│   ├── poincare.rs      # Poincare CUDA FFI wrapper (PATTERN TO FOLLOW)
│   ├── ops.rs           # GPU operation stubs
│   └── stub.rs          # Stub implementations
└── tests/
    └── cuda_poincare_test.rs   # Reference for test patterns
```

### Key Files You MUST Read Before Coding

| File | Purpose | Line Range |
|------|---------|------------|
| `crates/context-graph-cuda/src/poincare.rs` | FFI pattern to follow exactly | All |
| `crates/context-graph-cuda/src/error.rs` | Error types to use | All |
| `crates/context-graph-cuda/kernels/poincare_distance.cu` | CUDA kernel pattern | All |
| `crates/context-graph-graph/src/entailment/cones.rs` | **THE CANONICAL FORMULA** | Lines 272-352 |

### Git Status (Recent Commits)
- `4536e42` - M04-T22 standalone modulation utilities (COMPLETED)
- `11a4bb8` - M04-T21 contradiction detection (COMPLETED)
- `4fd5052` - M04-T20 entailment query (COMPLETED)
- `f891496` - M04-T19 domain-aware search (COMPLETED)
- `f044c84` - M04-T18 semantic search with FAISS GPU (COMPLETED)

---

## The CANONICAL Membership Score Formula

**THIS IS THE ONLY CORRECT FORMULA. DO NOT MODIFY.**

From `crates/context-graph-graph/src/entailment/cones.rs` lines 319-352:

```rust
/// CANONICAL FORMULA (DO NOT MODIFY)
///
/// - If angle <= effective_aperture: score = 1.0
/// - If angle > effective_aperture: score = exp(-2.0 * (angle - aperture))
pub fn membership_score(&self, point: &PoincarePoint, ball: &PoincareBall) -> f32 {
    let angle = self.compute_angle(point, ball);
    let aperture = self.effective_aperture();

    if angle <= aperture {
        1.0
    } else {
        (-2.0 * (angle - aperture)).exp()
    }
}
```

### Angle Computation Algorithm (from cones.rs lines 272-309)
```
1. tangent = log_map(apex, point) - direction to point in tangent space
2. to_origin = log_map(apex, origin) - cone axis direction (toward origin)
3. cos_angle = dot(tangent, to_origin) / (||tangent|| * ||to_origin||)
4. angle = acos(cos_angle.clamp(-1.0, 1.0))

Edge cases that return angle = 0.0:
- Point at apex (distance < eps)
- Apex at origin (norm < eps)
- Zero-length tangent or to_origin vectors
```

### Effective Aperture Formula (from cones.rs lines 204-213)
```rust
pub fn effective_aperture(&self) -> f32 {
    // aperture_factor scales with depth: deeper = narrower
    // base_aperture * factor gives the actual half-angle
    self.base_aperture * self.aperture_factor
}
```

---

## Hardware Targets (from constitution.yaml)

```yaml
stack:
  gpu: { target: "RTX 5090", vram: "32GB", compute: "12.0" }
  lang: { cuda: "13.1" }

perf:
  latency:
    entailment_check: "<1ms"
    cone_containment_gpu: "<2ms for 1K x 1K batch"
```

---

## Implementation Requirements

### 1. CUDA Kernel: `kernels/cone_check.cu`

**File must include these components:**

```cuda
// crates/context-graph-cuda/kernels/cone_check.cu

#include <cuda_runtime.h>
#include <math.h>
#include <stdio.h>

// ============================================================================
// KERNEL CONFIGURATION - DO NOT CHANGE
// ============================================================================
constexpr int BLOCK_DIM_X = 32;      // Warp-aligned (NVIDIA requirement)
constexpr int BLOCK_DIM_Y = 8;       // Threads per block Y dimension
constexpr int POINT_DIM = 64;        // Poincare ball dimension (fixed)
constexpr int CONE_DATA_DIM = 65;    // 64 apex coords + 1 aperture
constexpr int CONES_PER_BLOCK = 8;   // Cones cached in shared memory
constexpr float EPS = 1e-7f;         // Numerical stability epsilon

// ============================================================================
// DEVICE HELPER FUNCTIONS
// ============================================================================

/**
 * Compute Mobius addition: x (+) y in Poincare ball
 *
 * Formula: (x (+) y) = ((1 + 2c<x,y> + c||y||²)x + (1 - c||x||²)y) /
 *                      (1 + 2c<x,y> + c²||x||²||y||²)
 */
__device__ void mobius_add(
    const float* x,
    const float* y,
    float* result,
    float c
);

/**
 * Compute log map: log_x(y) - tangent vector at x pointing toward y
 *
 * This is the key operation for computing the angle between point and cone axis.
 */
__device__ void log_map(
    const float* x,      // Base point
    const float* y,      // Target point
    float* tangent,      // Output tangent vector
    float c              // Curvature magnitude (always positive)
);

/**
 * CANONICAL membership score formula
 *
 * 1. Compute tangent from apex to point
 * 2. Compute tangent from apex to origin (cone axis)
 * 3. Compute angle between tangents
 * 4. Apply score formula:
 *    - angle <= aperture: 1.0
 *    - angle > aperture: exp(-2.0 * (angle - aperture))
 */
__device__ float cone_membership_score(
    const float* apex,
    float aperture,
    const float* point,
    float c
);

// ============================================================================
// MAIN KERNEL
// ============================================================================

/**
 * Batch cone membership kernel
 *
 * Grid: ((n_points + 31) / 32, (n_cones + 7) / 8)
 * Block: (32, 8) = 256 threads
 * Shared: 8 * 65 * 4 = 2080 bytes per block
 *
 * @param cones      Cone data [n_cones][65] - row-major, 65 = 64 apex + 1 aperture
 * @param points     Point data [n_points][64] - row-major
 * @param scores     Output [n_cones][n_points] - row-major
 * @param n_cones    Number of cones
 * @param n_points   Number of points
 * @param curvature  Poincare ball curvature (negative, e.g., -1.0)
 */
__global__ void cone_check_kernel(
    const float* __restrict__ cones,
    const float* __restrict__ points,
    float* __restrict__ scores,
    int n_cones,
    int n_points,
    float curvature
);

// ============================================================================
// HOST LAUNCHER
// ============================================================================

/**
 * Launch cone check kernel from host code
 *
 * @return cudaSuccess (0) on success, CUDA error code on failure
 */
extern "C" int launch_cone_check(
    const float* d_cones,
    const float* d_points,
    float* d_scores,
    int n_cones,
    int n_points,
    float curvature,
    void* stream
);

/**
 * Get kernel configuration info for debugging
 */
extern "C" void get_cone_kernel_config(
    int* block_dim_x,
    int* block_dim_y,
    int* point_dim,
    int* cone_data_dim,
    int* shared_mem
);
```

### 2. Rust FFI: `src/cone.rs`

**Follow the pattern from `poincare.rs` exactly:**

```rust
// crates/context-graph-cuda/src/cone.rs

//! CUDA-accelerated entailment cone membership checking.
//!
//! # Performance
//! - GPU (RTX 5090): <2ms for 1K × 1K membership matrix
//! - CPU fallback: Uses context-graph-graph EntailmentCone
//!
//! # Constitution Reference
//! - TECH-GRAPH-004 Section 10.2: Cone CUDA Kernel
//! - perf.latency.entailment_check: <1ms

use std::ffi::c_void;
use crate::error::{CudaError, CudaResult};

// ============================================================================
// CONSTANTS
// ============================================================================

/// Cone data dimension (64 apex coords + 1 aperture)
pub const CONE_DATA_DIM: usize = 65;

/// Point dimension (must match POINCARE_DIM)
pub const POINT_DIM: usize = 64;

/// Default curvature (matches Poincare module)
pub const DEFAULT_CURVATURE: f32 = -1.0;

/// Numerical epsilon
pub const CONE_EPS: f32 = 1e-7;

// ============================================================================
// FFI DECLARATIONS
// ============================================================================

#[cfg(feature = "cuda")]
mod ffi {
    use std::ffi::c_void;
    use std::os::raw::c_int;

    #[link(name = "cone_check", kind = "static")]
    extern "C" {
        pub fn launch_cone_check(
            d_cones: *const f32,
            d_points: *const f32,
            d_scores: *mut f32,
            n_cones: c_int,
            n_points: c_int,
            curvature: f32,
            stream: *mut c_void,
        ) -> c_int;

        pub fn get_cone_kernel_config(
            block_dim_x: *mut c_int,
            block_dim_y: *mut c_int,
            point_dim: *mut c_int,
            cone_data_dim: *mut c_int,
            shared_mem: *mut c_int,
        );
    }
}

// ============================================================================
// CONFIGURATION
// ============================================================================

/// Configuration for cone CUDA operations.
#[derive(Debug, Clone)]
pub struct ConeCudaConfig {
    /// Curvature (must be negative, default -1.0).
    pub curvature: f32,
}

impl Default for ConeCudaConfig {
    fn default() -> Self {
        Self { curvature: DEFAULT_CURVATURE }
    }
}

impl ConeCudaConfig {
    /// Create config with custom curvature.
    pub fn with_curvature(curvature: f32) -> CudaResult<Self> {
        if curvature >= 0.0 {
            return Err(CudaError::InvalidConfig(
                "Cone curvature must be negative".to_string(),
            ));
        }
        if curvature.is_nan() {
            return Err(CudaError::InvalidConfig(
                "Cone curvature cannot be NaN".to_string(),
            ));
        }
        Ok(Self { curvature })
    }

    /// Validate configuration.
    pub fn validate(&self) -> CudaResult<()> {
        if self.curvature >= 0.0 {
            return Err(CudaError::InvalidConfig(
                "Cone curvature must be negative".to_string(),
            ));
        }
        if self.curvature.is_nan() {
            return Err(CudaError::InvalidConfig(
                "Cone curvature cannot be NaN".to_string(),
            ));
        }
        Ok(())
    }
}

// ============================================================================
// DATA STRUCTURES
// ============================================================================

/// Cone data packed for GPU transfer.
#[derive(Debug, Clone, Copy)]
pub struct ConeData {
    /// Apex point in Poincare ball (64 floats)
    pub apex: [f32; 64],
    /// Effective aperture angle in radians
    pub aperture: f32,
}

impl ConeData {
    /// Create new cone data.
    pub fn new(apex: [f32; 64], aperture: f32) -> CudaResult<Self> {
        // Validate apex is inside Poincare ball
        let norm_sq: f32 = apex.iter().map(|x| x * x).sum();
        if norm_sq >= 1.0 {
            return Err(CudaError::InvalidConfig(format!(
                "Apex norm {} >= 1.0, must be inside Poincare ball",
                norm_sq.sqrt()
            )));
        }
        if aperture < 0.0 {
            return Err(CudaError::InvalidConfig(
                "Aperture must be non-negative".to_string(),
            ));
        }
        Ok(Self { apex, aperture })
    }

    /// Pack to GPU format [apex_0..apex_63, aperture]
    pub fn to_gpu_format(&self) -> [f32; CONE_DATA_DIM] {
        let mut data = [0.0f32; CONE_DATA_DIM];
        data[..64].copy_from_slice(&self.apex);
        data[64] = self.aperture;
        data
    }

    /// Unpack from GPU format
    pub fn from_gpu_format(data: &[f32; CONE_DATA_DIM]) -> Self {
        let mut apex = [0.0f32; 64];
        apex.copy_from_slice(&data[..64]);
        Self {
            apex,
            aperture: data[64],
        }
    }
}

// ============================================================================
// GPU FUNCTIONS
// ============================================================================

/// Check if cone CUDA kernels are available.
#[cfg(feature = "cuda")]
pub fn is_cone_gpu_available() -> bool {
    extern "C" {
        fn cudaGetDeviceCount(count: *mut i32) -> i32;
    }
    unsafe {
        let mut device_count: i32 = 0;
        let result = cudaGetDeviceCount(&mut device_count);
        result == 0 && device_count > 0
    }
}

#[cfg(not(feature = "cuda"))]
pub fn is_cone_gpu_available() -> bool {
    false
}

/// Compute batch cone membership scores on GPU.
///
/// # Safety
/// - All device pointers must be valid and properly sized
/// - d_cones: [n_cones][65] floats
/// - d_points: [n_points][64] floats
/// - d_scores: [n_cones][n_points] floats (output)
#[cfg(feature = "cuda")]
pub unsafe fn cone_check_batch_gpu(
    d_cones: *const f32,
    d_points: *const f32,
    d_scores: *mut f32,
    n_cones: usize,
    n_points: usize,
    config: &ConeCudaConfig,
    stream: Option<*mut c_void>,
) -> CudaResult<()> {
    config.validate()?;

    if d_cones.is_null() {
        return Err(CudaError::InvalidConfig("d_cones pointer is null".to_string()));
    }
    if d_points.is_null() {
        return Err(CudaError::InvalidConfig("d_points pointer is null".to_string()));
    }
    if d_scores.is_null() {
        return Err(CudaError::InvalidConfig("d_scores pointer is null".to_string()));
    }
    if n_cones == 0 || n_points == 0 {
        return Err(CudaError::InvalidConfig(
            "n_cones and n_points must be positive".to_string(),
        ));
    }

    let stream_ptr = stream.unwrap_or(std::ptr::null_mut());

    let result = ffi::launch_cone_check(
        d_cones,
        d_points,
        d_scores,
        n_cones as i32,
        n_points as i32,
        config.curvature,
        stream_ptr,
    );

    if result != 0 {
        return Err(CudaError::KernelError(format!(
            "Cone check kernel failed with CUDA error code {}",
            result
        )));
    }

    Ok(())
}

#[cfg(not(feature = "cuda"))]
pub unsafe fn cone_check_batch_gpu(
    _d_cones: *const f32,
    _d_points: *const f32,
    _d_scores: *mut f32,
    _n_cones: usize,
    _n_points: usize,
    _config: &ConeCudaConfig,
    _stream: Option<*mut c_void>,
) -> CudaResult<()> {
    Err(CudaError::NotImplemented(
        "CUDA feature not enabled. Compile with --features cuda".to_string(),
    ))
}

// ============================================================================
// CPU REFERENCE IMPLEMENTATION
// ============================================================================

/// Compute single cone membership score on CPU.
///
/// CANONICAL FORMULA:
/// - If angle <= aperture: 1.0
/// - If angle > aperture: exp(-2.0 * (angle - aperture))
pub fn cone_membership_score_cpu(
    apex: &[f32; 64],
    aperture: f32,
    point: &[f32; 64],
    curvature: f32,
) -> f32 {
    let c = curvature.abs();

    // Compute tangent from apex to point
    let tangent = compute_log_map(apex, point, c);

    // Compute tangent from apex to origin
    let origin = [0.0f32; 64];
    let to_origin = compute_log_map(apex, &origin, c);

    // Compute angle between tangent vectors
    let dot: f32 = tangent.iter().zip(to_origin.iter()).map(|(a, b)| a * b).sum();
    let tangent_norm: f32 = tangent.iter().map(|x| x * x).sum::<f32>().sqrt();
    let origin_norm: f32 = to_origin.iter().map(|x| x * x).sum::<f32>().sqrt();

    // Handle degenerate cases
    if tangent_norm < CONE_EPS || origin_norm < CONE_EPS {
        return 1.0; // Point at apex or apex at origin
    }

    let cos_angle = (dot / (tangent_norm * origin_norm)).clamp(-1.0, 1.0);
    let angle = cos_angle.acos();

    // CANONICAL FORMULA
    if angle <= aperture {
        1.0
    } else {
        (-2.0 * (angle - aperture)).exp()
    }
}

/// Compute log map (simplified for CPU reference)
fn compute_log_map(base: &[f32; 64], target: &[f32; 64], c: f32) -> [f32; 64] {
    let mut tangent = [0.0f32; 64];

    // Compute Mobius subtraction direction
    let base_norm_sq: f32 = base.iter().map(|x| x * x).sum();
    let target_norm_sq: f32 = target.iter().map(|x| x * x).sum();

    let lambda_base = 2.0 / (1.0 - c * base_norm_sq).max(CONE_EPS);

    // Simplified direction computation
    let mut diff_norm_sq = 0.0f32;
    for i in 0..64 {
        tangent[i] = target[i] - base[i];
        diff_norm_sq += tangent[i] * tangent[i];
    }

    let diff_norm = diff_norm_sq.sqrt().max(CONE_EPS);

    // Normalize
    for i in 0..64 {
        tangent[i] /= diff_norm;
    }

    tangent
}

/// Compute batch cone membership scores on CPU.
pub fn cone_check_batch_cpu(
    cones: &[f32],      // [n_cones * 65]
    points: &[f32],     // [n_points * 64]
    n_cones: usize,
    n_points: usize,
    curvature: f32,
) -> Vec<f32> {
    assert_eq!(cones.len(), n_cones * CONE_DATA_DIM, "Invalid cones size");
    assert_eq!(points.len(), n_points * POINT_DIM, "Invalid points size");

    let mut scores = vec![0.0f32; n_cones * n_points];

    for i in 0..n_cones {
        let cone_start = i * CONE_DATA_DIM;
        let apex: &[f32; 64] = cones[cone_start..cone_start + 64].try_into().unwrap();
        let aperture = cones[cone_start + 64];

        for j in 0..n_points {
            let pt_start = j * POINT_DIM;
            let point: &[f32; 64] = points[pt_start..pt_start + 64].try_into().unwrap();

            scores[i * n_points + j] = cone_membership_score_cpu(apex, aperture, point, curvature);
        }
    }

    scores
}

// ============================================================================
// KERNEL INFO
// ============================================================================

/// Kernel configuration info for debugging.
#[derive(Debug, Clone, Copy)]
pub struct ConeKernelInfo {
    pub block_dim_x: i32,
    pub block_dim_y: i32,
    pub point_dim: i32,
    pub cone_data_dim: i32,
    pub shared_mem_bytes: i32,
}

#[cfg(feature = "cuda")]
pub fn get_cone_kernel_info() -> Option<ConeKernelInfo> {
    let mut block_dim_x: i32 = 0;
    let mut block_dim_y: i32 = 0;
    let mut point_dim: i32 = 0;
    let mut cone_data_dim: i32 = 0;
    let mut shared_mem: i32 = 0;

    unsafe {
        ffi::get_cone_kernel_config(
            &mut block_dim_x,
            &mut block_dim_y,
            &mut point_dim,
            &mut cone_data_dim,
            &mut shared_mem,
        );
    }

    Some(ConeKernelInfo {
        block_dim_x,
        block_dim_y,
        point_dim,
        cone_data_dim,
        shared_mem_bytes: shared_mem,
    })
}

#[cfg(not(feature = "cuda"))]
pub fn get_cone_kernel_info() -> Option<ConeKernelInfo> {
    None
}
```

### 3. Update `src/lib.rs`

Add after the poincare export:
```rust
pub mod cone;
pub use cone::{
    ConeData, ConeCudaConfig, ConeKernelInfo,
    cone_check_batch_gpu, cone_check_batch_cpu, cone_membership_score_cpu,
    is_cone_gpu_available, get_cone_kernel_info,
    CONE_DATA_DIM, POINT_DIM,
};
```

### 4. Update `build.rs`

```rust
// crates/context-graph-cuda/build.rs

fn main() {
    #[cfg(feature = "cuda")]
    {
        compile_cuda_kernels();
    }
}

#[cfg(feature = "cuda")]
fn compile_cuda_kernels() {
    use std::env;
    use std::path::PathBuf;

    let cuda_path = env::var("CUDA_PATH").unwrap_or_else(|_| "/usr/local/cuda".to_string());

    println!("cargo:rerun-if-changed=kernels/poincare_distance.cu");
    println!("cargo:rerun-if-changed=kernels/cone_check.cu");

    // Compile Poincare distance kernel (existing)
    cc::Build::new()
        .cuda(true)
        .cudart("static")
        .flag("-arch=sm_90")  // Compute 12.0 for RTX 5090
        .flag("-O3")
        .include(format!("{}/include", cuda_path))
        .file("kernels/poincare_distance.cu")
        .compile("poincare_distance");

    // Compile Cone check kernel (NEW)
    cc::Build::new()
        .cuda(true)
        .cudart("static")
        .flag("-arch=sm_90")
        .flag("-O3")
        .include(format!("{}/include", cuda_path))
        .file("kernels/cone_check.cu")
        .compile("cone_check");

    println!("cargo:rustc-link-search=native={}/lib64", cuda_path);
    println!("cargo:rustc-link-lib=cudart");
}
```

### 5. Update `Cargo.toml`

Add cc as build dependency:
```toml
[build-dependencies]
cc = { version = "1.0", features = ["parallel"] }
```

---

## Test Requirements

**Location:** `crates/context-graph-cuda/tests/cuda_cone_test.rs`

**Required Tests (NO MOCKS - REAL DATA ONLY):**

```rust
//! Integration tests for Cone CUDA operations.
//! NO MOCKS - REAL DATA ONLY per constitution REQ-KG-TEST.

use context_graph_cuda::cone::{
    cone_membership_score_cpu, cone_check_batch_cpu, is_cone_gpu_available,
    ConeCudaConfig, ConeData, CONE_DATA_DIM, POINT_DIM,
};
use context_graph_cuda::CudaError;

// ============================================================================
// CONFIGURATION TESTS
// ============================================================================

#[test]
fn test_config_defaults_are_valid() {
    let config = ConeCudaConfig::default();
    assert!(config.validate().is_ok());
    assert!((config.curvature - (-1.0)).abs() < 1e-6);
}

#[test]
fn test_config_rejects_positive_curvature() {
    let result = ConeCudaConfig::with_curvature(0.5);
    assert!(result.is_err());
}

#[test]
fn test_config_rejects_nan_curvature() {
    let result = ConeCudaConfig::with_curvature(f32::NAN);
    assert!(result.is_err());
}

// ============================================================================
// CPU REFERENCE TESTS
// ============================================================================

#[test]
fn test_cpu_score_point_at_apex_returns_1() {
    let apex = [0.1f32; 64];
    let point = apex.clone();  // Same as apex
    let score = cone_membership_score_cpu(&apex, 0.5, &point, -1.0);
    assert!((score - 1.0).abs() < 1e-4, "Point at apex should have score 1.0");
}

#[test]
fn test_cpu_score_inside_cone_returns_1() {
    // Apex near origin, point also near origin = small angle = inside cone
    let apex = [0.1f32; 64];
    let mut point = [0.05f32; 64];
    point[0] = 0.05;  // Slightly different
    let score = cone_membership_score_cpu(&apex, 1.5, &point, -1.0);  // Wide aperture
    assert!(score > 0.9, "Point inside wide cone should have high score: {}", score);
}

#[test]
fn test_cpu_score_outside_cone_decays() {
    // Create a narrow cone and a point far outside
    let mut apex = [0.0f32; 64];
    apex[0] = 0.5;
    let aperture = 0.1;  // Very narrow

    let mut point = [0.0f32; 64];
    point[1] = 0.5;  // Perpendicular direction

    let score = cone_membership_score_cpu(&apex, aperture, &point, -1.0);
    assert!(score < 0.5, "Point outside narrow cone should have low score: {}", score);
}

#[test]
fn test_cpu_score_canonical_formula_verified() {
    // Manually verify the formula: exp(-2 * (angle - aperture))
    let apex = [0.0f32; 64];
    let aperture = 0.5;

    // Point at angle ~0.7 from cone axis
    let mut point = [0.0f32; 64];
    point[0] = 0.3;
    point[1] = 0.3;

    let score = cone_membership_score_cpu(&apex, aperture, &point, -1.0);

    // Score should be exp(-2 * (angle - 0.5)) if angle > 0.5
    // Since angle > aperture, score should be < 1.0
    println!("CANONICAL FORMULA TEST: aperture={}, score={}", aperture, score);
    assert!(score >= 0.0 && score <= 1.0, "Score must be in [0,1]");
}

// ============================================================================
// BATCH TESTS
// ============================================================================

#[test]
fn test_batch_cpu_matches_single() {
    let n_cones = 5;
    let n_points = 5;

    let cones: Vec<f32> = (0..n_cones)
        .flat_map(|i| {
            let mut data = [0.0f32; 65];
            data[0] = (i as f32) * 0.1;  // Varying apex
            data[64] = 0.5;  // Aperture
            data.to_vec()
        })
        .collect();

    let points: Vec<f32> = (0..n_points)
        .flat_map(|i| {
            let mut p = [0.0f32; 64];
            p[0] = (i as f32) * 0.1 + 0.05;
            p.to_vec()
        })
        .collect();

    let batch_scores = cone_check_batch_cpu(&cones, &points, n_cones, n_points, -1.0);

    assert_eq!(batch_scores.len(), n_cones * n_points);

    // Verify batch matches single computations
    for i in 0..n_cones {
        let apex: &[f32; 64] = cones[i * 65..i * 65 + 64].try_into().unwrap();
        let aperture = cones[i * 65 + 64];

        for j in 0..n_points {
            let point: &[f32; 64] = points[j * 64..(j + 1) * 64].try_into().unwrap();
            let single = cone_membership_score_cpu(apex, aperture, point, -1.0);
            let batch = batch_scores[i * n_points + j];

            assert!(
                (single - batch).abs() < 1e-5,
                "Mismatch at [{}, {}]: single={}, batch={}",
                i, j, single, batch
            );
        }
    }
}

#[test]
fn test_batch_cpu_100x100() {
    let n = 100;

    let cones: Vec<f32> = (0..n)
        .flat_map(|i| {
            let mut data = [0.0f32; 65];
            data[0] = ((i * 17) % 80) as f32 * 0.01;
            data[64] = 0.5 + (i % 10) as f32 * 0.05;
            data.to_vec()
        })
        .collect();

    let points: Vec<f32> = (0..n)
        .flat_map(|i| {
            let mut p = [0.0f32; 64];
            p[0] = ((i * 23) % 90) as f32 * 0.01;
            p.to_vec()
        })
        .collect();

    let scores = cone_check_batch_cpu(&cones, &points, n, n, -1.0);

    assert_eq!(scores.len(), n * n);

    // All scores must be valid
    for (idx, &s) in scores.iter().enumerate() {
        assert!(s >= 0.0 && s <= 1.0 && s.is_finite(),
            "Invalid score at {}: {}", idx, s);
    }
}

// ============================================================================
// EDGE CASES
// ============================================================================

#[test]
fn test_edge_case_apex_at_origin() {
    let apex = [0.0f32; 64];
    let mut point = [0.0f32; 64];
    point[0] = 0.5;

    let score = cone_membership_score_cpu(&apex, 0.5, &point, -1.0);
    assert!(score.is_finite(), "Score should be finite for apex at origin");
}

#[test]
fn test_edge_case_boundary_point() {
    let apex = [0.1f32; 64];
    let scale = 0.99 / (64.0_f32).sqrt();
    let boundary_point: [f32; 64] = [scale; 64];

    let score = cone_membership_score_cpu(&apex, 0.5, &boundary_point, -1.0);
    assert!(score.is_finite() && score >= 0.0 && score <= 1.0,
        "Boundary point score invalid: {}", score);
}

#[test]
fn test_edge_case_zero_aperture() {
    let apex = [0.1f32; 64];
    let mut point = [0.0f32; 64];
    point[0] = 0.2;

    let score = cone_membership_score_cpu(&apex, 0.0, &point, -1.0);
    // With zero aperture, only apex should score 1.0
    assert!(score < 1.0, "Non-apex point with zero aperture should have score < 1.0");
}

// ============================================================================
// GPU AVAILABILITY TEST
// ============================================================================

#[test]
fn test_gpu_availability_check() {
    let available = is_cone_gpu_available();
    println!("Cone GPU available: {}", available);

    #[cfg(not(feature = "cuda"))]
    assert!(!available, "Without cuda feature, GPU should not be available");
}

// ============================================================================
// CONSTANTS VERIFICATION
// ============================================================================

#[test]
fn test_constants() {
    assert_eq!(CONE_DATA_DIM, 65, "CONE_DATA_DIM should be 65");
    assert_eq!(POINT_DIM, 64, "POINT_DIM should be 64");
}
```

---

## Full State Verification Requirements

### Source of Truth Definition
After kernel execution, the `scores` output buffer contains membership scores in GPU memory at address `d_scores`. This is the Source of Truth.

### Verification Protocol

**1. Execute & Inspect:**
```rust
// After kernel launch, copy results to host and verify
let mut host_scores = vec![0.0f32; n_cones * n_points];
cuda_memcpy_d2h(&mut host_scores, d_scores);

// Verify specific known values
assert!((host_scores[0] - expected_score_0).abs() < 1e-5);
```

**2. Boundary & Edge Case Audit (REQUIRED - Print Before/After State):**

**Edge Case 1: Empty inputs**
```rust
// BEFORE: n_cones = 0, n_points = 100
println!("[CONE_TEST] BEFORE: n_cones={}, n_points={}", 0, 100);
let result = cone_check_batch_gpu(...);
println!("[CONE_TEST] AFTER: result={:?}", result);
// EXPECTED: Returns error, not crash
```

**Edge Case 2: Maximum limits (1K x 1K)**
```rust
// BEFORE: n_cones = 1000, n_points = 1000
println!("[CONE_TEST] BEFORE: n_cones={}, n_points={}, allocating GPU memory", 1000, 1000);
let start = Instant::now();
cone_check_batch_gpu(...)?;
let elapsed = start.elapsed();
println!("[CONE_TEST] AFTER: elapsed={:?}ms, target <2ms", elapsed.as_millis());
assert!(elapsed.as_millis() < 2, "Performance target missed: {}ms", elapsed.as_millis());
```

**Edge Case 3: Invalid aperture (negative)**
```rust
// BEFORE: cone.aperture = -0.5 (invalid)
println!("[CONE_TEST] BEFORE: aperture={} (invalid)", -0.5);
let result = ConeData::new(apex, -0.5);
println!("[CONE_TEST] AFTER: result={:?}", result);
// EXPECTED: Returns error
```

**3. Evidence of Success (REQUIRED LOG OUTPUT):**
```
[CONE_CUDA_TEST] ===== VERIFICATION LOG =====
[CONE_CUDA_TEST] Test: test_gpu_matches_cpu_reference
[CONE_CUDA_TEST] n_cones: 100, n_points: 100
[CONE_CUDA_TEST] GPU execution time: 0.85ms
[CONE_CUDA_TEST] Sample scores (GPU): [1.0, 0.95, 0.82, 0.67, ...]
[CONE_CUDA_TEST] Sample scores (CPU): [1.0, 0.95, 0.82, 0.67, ...]
[CONE_CUDA_TEST] Max difference: 3.2e-6
[CONE_CUDA_TEST] All 10000 scores verified within 1e-5 tolerance
[CONE_CUDA_TEST] PASS
```

---

## Acceptance Criteria

- [ ] CUDA kernel compiles with nvcc (compute capability 9.0 for RTX 5090)
- [ ] Shared memory used for cone data (2080 bytes: 8 * 65 * 4)
- [ ] Matches CPU implementation within 1e-5 tolerance
- [ ] Performance: <2ms for 1K x 1K membership matrix
- [ ] Returns soft membership score [0,1]
- [ ] Compiles with `cargo build -p context-graph-cuda`
- [ ] Tests pass with `cargo test -p context-graph-cuda`
- [ ] No clippy warnings
- [ ] Error messages include context for debugging
- [ ] All edge cases handled without panic

---

## Verification Commands

```bash
# 1. Check CUDA is available
nvcc --version
# Expected: CUDA 13.x or at least 12.0

# 2. Build the crate (CPU-only, no CUDA)
cargo build -p context-graph-cuda
# Expected: Compiles without error

# 3. Build with CUDA feature
cargo build -p context-graph-cuda --features cuda
# Expected: Compiles kernels without error

# 4. Run CPU tests (no GPU required)
cargo test -p context-graph-cuda -- test_cpu --nocapture
# Expected: All pass with logs visible

# 5. Run GPU tests (requires CUDA device)
cargo test -p context-graph-cuda --features cuda -- test_gpu --nocapture
# Expected: All pass, <2ms performance

# 6. Run full test suite
cargo test -p context-graph-cuda --features cuda -- --nocapture
# Expected: All pass with verification logs visible

# 7. Check clippy
cargo clippy -p context-graph-cuda -- -D warnings
# Expected: No warnings
```

---

## FINAL VERIFICATION STEP

**MANDATORY: Use sherlock-holmes subagent to verify task completion**

After you believe the implementation is complete:

1. Spawn sherlock-holmes agent with prompt:
   ```
   Forensically investigate M04-T24 cone CUDA kernel implementation.

   VERIFY BY CHECKING ACTUAL FILES AND RUNNING COMMANDS:
   1. File exists: crates/context-graph-cuda/kernels/cone_check.cu
   2. File exists: crates/context-graph-cuda/src/cone.rs
   3. cone module exported in lib.rs (grep for "pub mod cone")
   4. build.rs compiles cone_check.cu (grep for "cone_check")
   5. Tests exist in tests/cuda_cone_test.rs
   6. Tests use REAL DATA, not mocks (grep for "mock" should return empty)
   7. Canonical formula matches cones.rs: exp(-2*(angle-aperture))
   8. All tests pass: cargo test -p context-graph-cuda
   9. No clippy warnings: cargo clippy -p context-graph-cuda
   10. Build succeeds: cargo build -p context-graph-cuda

   For each check:
   - Run the verification command
   - Print the output
   - Mark PASS or FAIL

   Report any discrepancies as CRITICAL issues.
   ```

2. Fix any issues identified by sherlock-holmes
3. Re-run sherlock-holmes until all checks pass

---

## References

- [cudarc - Rust CUDA wrapper](https://github.com/coreylowman/cudarc)
- [bindgen_cuda - CUDA kernel compilation](https://docs.rs/bindgen_cuda/latest/bindgen_cuda/)
- [Hyperbolic Entailment Cones Paper](https://arxiv.org/abs/1804.01882)
- [Original Implementation (Python/TensorFlow)](https://github.com/dalab/hyperbolic_cones)
- [Rust CUDA 2025 Update](https://rust-gpu.github.io/blog/2025/05/27/rust-cuda-update/)

---

*Task Version: 2.0.0*
*Last Updated: 2026-01-04*
*Audited Against: commit 4536e42*
