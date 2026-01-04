//! CUDA-accelerated Poincare ball distance computation.
//!
//! Provides GPU-accelerated batch distance computation for hyperbolic geometry.
//! Falls back to CPU when GPU is unavailable.
//!
//! # Performance
//!
//! - GPU (RTX 5090): <1ms for 1K × 1K distance matrix
//! - CPU fallback: Uses context-graph-graph PoincareBall
//!
//! # Constitution Reference
//!
//! - TECH-GRAPH-004 Section 10.1: Poincare CUDA Kernel
//! - perf.latency.poincare_distance_gpu: <1ms for 1K × 1K
//! - stack.gpu: RTX 5090, 32GB GDDR7, compute: 12.0
//!
//! # Mathematics
//!
//! Poincare ball distance formula (direct, GPU-friendly):
//! ```text
//! d(x,y) = (2/sqrt(c)) * arctanh(sqrt(c * ||x-y||² / ((1-c*||x||²)(1-c*||y||²))))
//! ```
//!
//! where c = |curvature| (always positive for hyperbolic space).

use std::ffi::c_void;

use crate::error::{CudaError, CudaResult};

/// Default Poincare ball dimension (fixed for SIMD alignment).
/// Must match the kernel's POINT_DIM constant.
pub const POINCARE_DIM: usize = 64;

/// Default curvature (negative, per hyperbolic geometry).
/// Standard unit hyperbolic space has curvature -1.0.
pub const DEFAULT_CURVATURE: f32 = -1.0;

/// Numerical epsilon for stability (matches kernel).
pub const POINCARE_EPS: f32 = 1e-7;

// ============================================================================
// FFI Declarations (CUDA kernels)
// ============================================================================

/// FFI declarations for CUDA kernel functions.
/// Only available when the `cuda` feature is enabled.
#[cfg(feature = "cuda")]
mod ffi {
    use std::ffi::c_void;
    use std::os::raw::c_int;

    #[link(name = "poincare_distance", kind = "static")]
    extern "C" {
        /// Launch batch Poincare distance computation.
        pub fn launch_poincare_distance(
            d_queries: *const f32,
            d_database: *const f32,
            d_distances: *mut f32,
            n_queries: c_int,
            n_database: c_int,
            curvature: f32,
            stream: *mut c_void,
        ) -> c_int;

        /// Single-pair distance (delegates to batch with n=1).
        pub fn poincare_distance_single(
            d_point_a: *const f32,
            d_point_b: *const f32,
            d_distance: *mut f32,
            curvature: f32,
            stream: *mut c_void,
        ) -> c_int;

        /// Get kernel configuration info.
        pub fn get_poincare_kernel_config(
            block_dim_x: *mut c_int,
            block_dim_y: *mut c_int,
            point_dim: *mut c_int,
            shared_mem: *mut c_int,
        );
    }
}

// ============================================================================
// Configuration
// ============================================================================

/// Configuration for Poincare CUDA operations.
///
/// # Example
///
/// ```
/// use context_graph_cuda::poincare::PoincareCudaConfig;
///
/// let config = PoincareCudaConfig::default();
/// assert_eq!(config.dim, 64);
/// assert!((config.curvature - (-1.0)).abs() < 1e-6);
/// assert!(config.validate().is_ok());
/// ```
#[derive(Debug, Clone)]
pub struct PoincareCudaConfig {
    /// Dimension of Poincare ball (must be 64 for CUDA kernel).
    pub dim: usize,
    /// Curvature (must be negative, default -1.0).
    pub curvature: f32,
}

impl Default for PoincareCudaConfig {
    fn default() -> Self {
        Self {
            dim: POINCARE_DIM,
            curvature: DEFAULT_CURVATURE,
        }
    }
}

impl PoincareCudaConfig {
    /// Create config with custom curvature.
    ///
    /// # Errors
    ///
    /// Returns error if curvature is not negative.
    ///
    /// # Example
    ///
    /// ```
    /// use context_graph_cuda::poincare::PoincareCudaConfig;
    ///
    /// let config = PoincareCudaConfig::with_curvature(-0.5).unwrap();
    /// assert!((config.curvature - (-0.5)).abs() < 1e-6);
    ///
    /// // Invalid curvature returns error
    /// assert!(PoincareCudaConfig::with_curvature(0.5).is_err());
    /// ```
    pub fn with_curvature(curvature: f32) -> CudaResult<Self> {
        if curvature >= 0.0 {
            return Err(CudaError::InvalidConfig(
                "Poincare curvature must be negative".to_string(),
            ));
        }
        if curvature.is_nan() {
            return Err(CudaError::InvalidConfig(
                "Poincare curvature cannot be NaN".to_string(),
            ));
        }
        Ok(Self {
            dim: POINCARE_DIM,
            curvature,
        })
    }

    /// Create config with custom dimension and curvature.
    ///
    /// # Errors
    ///
    /// Returns error if dimension is not 64 or curvature is not negative.
    pub fn with_dim_and_curvature(dim: usize, curvature: f32) -> CudaResult<Self> {
        if dim != POINCARE_DIM {
            return Err(CudaError::InvalidConfig(format!(
                "Poincare dimension must be {} for CUDA kernel, got {}",
                POINCARE_DIM, dim
            )));
        }
        Self::with_curvature(curvature)
    }

    /// Validate configuration parameters.
    ///
    /// # Errors
    ///
    /// Returns error if:
    /// - Dimension is not 64
    /// - Curvature is not negative
    /// - Curvature is NaN
    ///
    /// # Example
    ///
    /// ```
    /// use context_graph_cuda::poincare::PoincareCudaConfig;
    ///
    /// let config = PoincareCudaConfig::default();
    /// assert!(config.validate().is_ok());
    ///
    /// let bad_config = PoincareCudaConfig { dim: 128, curvature: -1.0 };
    /// assert!(bad_config.validate().is_err());
    /// ```
    pub fn validate(&self) -> CudaResult<()> {
        if self.dim != POINCARE_DIM {
            return Err(CudaError::InvalidConfig(format!(
                "Poincare dimension must be {} for CUDA kernel, got {}",
                POINCARE_DIM, self.dim
            )));
        }
        if self.curvature >= 0.0 {
            return Err(CudaError::InvalidConfig(
                "Poincare curvature must be negative".to_string(),
            ));
        }
        if self.curvature.is_nan() {
            return Err(CudaError::InvalidConfig(
                "Poincare curvature cannot be NaN".to_string(),
            ));
        }
        Ok(())
    }

    /// Get absolute value of curvature (always positive).
    #[inline]
    pub fn abs_curvature(&self) -> f32 {
        self.curvature.abs()
    }
}

// ============================================================================
// Kernel Info
// ============================================================================

/// Information about the CUDA kernel configuration.
///
/// Useful for debugging and performance tuning.
#[derive(Debug, Clone, Copy)]
pub struct PoincareKernelInfo {
    /// Block dimension X (warp-aligned, typically 32)
    pub block_dim_x: i32,
    /// Block dimension Y (queries per block, typically 8)
    pub block_dim_y: i32,
    /// Point dimension (must be 64)
    pub point_dim: i32,
    /// Shared memory per block in bytes
    pub shared_mem_bytes: i32,
}

/// Get kernel configuration info.
///
/// Returns None if CUDA feature is disabled.
///
/// # Example
///
/// ```
/// use context_graph_cuda::poincare::get_kernel_info;
///
/// if let Some(info) = get_kernel_info() {
///     println!("Block size: {}x{}", info.block_dim_x, info.block_dim_y);
///     println!("Shared memory: {} bytes", info.shared_mem_bytes);
/// }
/// ```
#[cfg(feature = "cuda")]
pub fn get_kernel_info() -> Option<PoincareKernelInfo> {
    let mut block_dim_x: i32 = 0;
    let mut block_dim_y: i32 = 0;
    let mut point_dim: i32 = 0;
    let mut shared_mem: i32 = 0;

    unsafe {
        ffi::get_poincare_kernel_config(
            &mut block_dim_x,
            &mut block_dim_y,
            &mut point_dim,
            &mut shared_mem,
        );
    }

    Some(PoincareKernelInfo {
        block_dim_x,
        block_dim_y,
        point_dim,
        shared_mem_bytes: shared_mem,
    })
}

#[cfg(not(feature = "cuda"))]
pub fn get_kernel_info() -> Option<PoincareKernelInfo> {
    None
}

// ============================================================================
// GPU Availability
// ============================================================================

/// Check if CUDA Poincare kernels are available.
///
/// Returns true if:
/// 1. The `cuda` feature is enabled at compile time
/// 2. The system has a CUDA-capable GPU
/// 3. The CUDA runtime is available
///
/// # Example
///
/// ```
/// use context_graph_cuda::poincare::is_poincare_gpu_available;
///
/// if is_poincare_gpu_available() {
///     println!("GPU acceleration available!");
/// } else {
///     println!("Using CPU fallback");
/// }
/// ```
#[cfg(feature = "cuda")]
pub fn is_poincare_gpu_available() -> bool {
    // Try to query CUDA device count
    // This will fail if no CUDA runtime is available
    extern "C" {
        fn cudaGetDeviceCount(count: *mut i32) -> i32;
    }

    unsafe {
        let mut device_count: i32 = 0;
        let result = cudaGetDeviceCount(&mut device_count);
        // cudaSuccess = 0
        result == 0 && device_count > 0
    }
}

#[cfg(not(feature = "cuda"))]
pub fn is_poincare_gpu_available() -> bool {
    false
}

// ============================================================================
// GPU Functions
// ============================================================================

/// Compute batch Poincare distances on GPU.
///
/// # Safety
///
/// - `d_queries`, `d_database`, `d_distances` must be valid device pointers
/// - Arrays must be properly sized: queries\[n_queries\]\[64\], database\[n_database\]\[64\]
/// - Output distances\[n_queries\]\[n_database\] must be pre-allocated
/// - Pointers must be aligned for float32 access
///
/// # Arguments
///
/// * `d_queries` - Device pointer to query vectors \[n_queries\]\[64\]
/// * `d_database` - Device pointer to database vectors \[n_database\]\[64\]
/// * `d_distances` - Device pointer to output matrix \[n_queries\]\[n_database\]
/// * `n_queries` - Number of query points
/// * `n_database` - Number of database points
/// * `config` - Poincare configuration
/// * `stream` - CUDA stream (None for default stream)
///
/// # Errors
///
/// Returns `CudaError` if:
/// - Configuration is invalid
/// - Kernel launch fails
/// - CUDA runtime error occurs
///
/// # Performance
///
/// Target: <1ms for 1K × 1K distance matrix on RTX 5090.
///
/// # Example
///
/// ```ignore
/// use context_graph_cuda::poincare::{poincare_distance_batch_gpu, PoincareCudaConfig};
///
/// // Assume d_queries, d_database, d_distances are valid device pointers
/// unsafe {
///     let config = PoincareCudaConfig::default();
///     poincare_distance_batch_gpu(
///         d_queries, d_database, d_distances,
///         1000, 1000, &config, None
///     )?;
/// }
/// ```
#[cfg(feature = "cuda")]
pub unsafe fn poincare_distance_batch_gpu(
    d_queries: *const f32,
    d_database: *const f32,
    d_distances: *mut f32,
    n_queries: usize,
    n_database: usize,
    config: &PoincareCudaConfig,
    stream: Option<*mut c_void>,
) -> CudaResult<()> {
    // Validate configuration
    config.validate()?;

    // Validate pointer arguments
    if d_queries.is_null() {
        return Err(CudaError::InvalidConfig(
            "d_queries pointer is null".to_string(),
        ));
    }
    if d_database.is_null() {
        return Err(CudaError::InvalidConfig(
            "d_database pointer is null".to_string(),
        ));
    }
    if d_distances.is_null() {
        return Err(CudaError::InvalidConfig(
            "d_distances pointer is null".to_string(),
        ));
    }

    // Validate sizes
    if n_queries == 0 || n_database == 0 {
        return Err(CudaError::InvalidConfig(
            "n_queries and n_database must be positive".to_string(),
        ));
    }

    let stream_ptr = stream.unwrap_or(std::ptr::null_mut());

    let result = ffi::launch_poincare_distance(
        d_queries,
        d_database,
        d_distances,
        n_queries as i32,
        n_database as i32,
        config.curvature,
        stream_ptr,
    );

    if result != 0 {
        return Err(CudaError::KernelError(format!(
            "Poincare distance kernel failed with CUDA error code {}",
            result
        )));
    }

    Ok(())
}

/// Stub implementation when CUDA feature is disabled.
///
/// # Safety
///
/// This stub always returns an error and does not use the pointers.
#[cfg(not(feature = "cuda"))]
pub unsafe fn poincare_distance_batch_gpu(
    _d_queries: *const f32,
    _d_database: *const f32,
    _d_distances: *mut f32,
    _n_queries: usize,
    _n_database: usize,
    _config: &PoincareCudaConfig,
    _stream: Option<*mut c_void>,
) -> CudaResult<()> {
    Err(CudaError::NotImplemented(
        "CUDA feature not enabled. Compile with --features cuda".to_string(),
    ))
}

/// Compute single-pair Poincare distance on GPU.
///
/// # Safety
///
/// Same requirements as `poincare_distance_batch_gpu`.
///
/// # Note
///
/// For efficiency, prefer `poincare_distance_batch_gpu` when computing
/// multiple distances. Single-pair calls have significant kernel launch overhead.
#[cfg(feature = "cuda")]
pub unsafe fn poincare_distance_single_gpu(
    d_point_a: *const f32,
    d_point_b: *const f32,
    d_distance: *mut f32,
    config: &PoincareCudaConfig,
    stream: Option<*mut c_void>,
) -> CudaResult<()> {
    config.validate()?;

    if d_point_a.is_null() || d_point_b.is_null() || d_distance.is_null() {
        return Err(CudaError::InvalidConfig(
            "Device pointers cannot be null".to_string(),
        ));
    }

    let stream_ptr = stream.unwrap_or(std::ptr::null_mut());

    let result = ffi::poincare_distance_single(
        d_point_a,
        d_point_b,
        d_distance,
        config.curvature,
        stream_ptr,
    );

    if result != 0 {
        return Err(CudaError::KernelError(format!(
            "Poincare distance single kernel failed with CUDA error code {}",
            result
        )));
    }

    Ok(())
}

/// Stub implementation when CUDA feature is disabled.
///
/// # Safety
///
/// This stub always returns an error and does not use the pointers.
#[cfg(not(feature = "cuda"))]
pub unsafe fn poincare_distance_single_gpu(
    _d_point_a: *const f32,
    _d_point_b: *const f32,
    _d_distance: *mut f32,
    _config: &PoincareCudaConfig,
    _stream: Option<*mut c_void>,
) -> CudaResult<()> {
    Err(CudaError::NotImplemented(
        "CUDA feature not enabled. Compile with --features cuda".to_string(),
    ))
}

// ============================================================================
// CPU Reference Implementation (for testing and fallback)
// ============================================================================

/// Compute Poincare distance on CPU (reference implementation).
///
/// This is the direct formula implementation, mathematically equivalent to
/// the GPU kernel. Used for testing and as a fallback when GPU is unavailable.
///
/// # Formula
///
/// ```text
/// d(x,y) = (2/sqrt(c)) * arctanh(sqrt(c * ||x-y||² / ((1-c*||x||²)(1-c*||y||²))))
/// ```
///
/// # Arguments
///
/// * `x` - First point (64-element array)
/// * `y` - Second point (64-element array)
/// * `curvature` - Poincare ball curvature (must be negative)
///
/// # Returns
///
/// Hyperbolic distance (always >= 0).
///
/// # Example
///
/// ```
/// use context_graph_cuda::poincare::poincare_distance_cpu;
///
/// let x = [0.0f32; 64];
/// let mut y = [0.0f32; 64];
/// y[0] = 0.5;
///
/// let dist = poincare_distance_cpu(&x, &y, -1.0);
/// assert!(dist > 0.0);
///
/// // Distance to self is zero
/// let dist_self = poincare_distance_cpu(&x, &x, -1.0);
/// assert!(dist_self < 1e-6);
/// ```
pub fn poincare_distance_cpu(x: &[f32; 64], y: &[f32; 64], curvature: f32) -> f32 {
    let c = curvature.abs();
    let sqrt_c = c.sqrt();

    // Compute norms
    let x_norm_sq: f32 = x.iter().map(|v| v * v).sum();
    let y_norm_sq: f32 = y.iter().map(|v| v * v).sum();

    // Compute ||x - y||²
    let diff_norm_sq: f32 = x
        .iter()
        .zip(y.iter())
        .map(|(xi, yi)| (xi - yi) * (xi - yi))
        .sum();

    // Handle identical points
    if diff_norm_sq < POINCARE_EPS {
        return 0.0;
    }

    // Denominators with numerical stability
    let denom_x = (1.0 - c * x_norm_sq).max(POINCARE_EPS);
    let denom_y = (1.0 - c * y_norm_sq).max(POINCARE_EPS);

    // arctanh argument
    let arg_sq = c * diff_norm_sq / (denom_x * denom_y);
    let arg = arg_sq.max(0.0).sqrt().min(1.0 - POINCARE_EPS);

    // Poincare distance
    (2.0 / sqrt_c) * arg.atanh()
}

/// Compute batch Poincare distances on CPU.
///
/// Reference implementation for testing GPU kernel correctness.
///
/// # Arguments
///
/// * `queries` - Query vectors, flattened \[n_queries * 64\]
/// * `database` - Database vectors, flattened \[n_database * 64\]
/// * `n_queries` - Number of query points
/// * `n_database` - Number of database points
/// * `curvature` - Poincare ball curvature (must be negative)
///
/// # Returns
///
/// Distance matrix, flattened \[n_queries * n_database\], row-major.
pub fn poincare_distance_batch_cpu(
    queries: &[f32],
    database: &[f32],
    n_queries: usize,
    n_database: usize,
    curvature: f32,
) -> Vec<f32> {
    assert_eq!(queries.len(), n_queries * 64, "Invalid query array size");
    assert_eq!(
        database.len(),
        n_database * 64,
        "Invalid database array size"
    );

    let mut distances = vec![0.0f32; n_queries * n_database];

    for i in 0..n_queries {
        let q_start = i * 64;
        let q: &[f32; 64] = queries[q_start..q_start + 64].try_into().unwrap();

        for j in 0..n_database {
            let db_start = j * 64;
            let db: &[f32; 64] = database[db_start..db_start + 64].try_into().unwrap();

            distances[i * n_database + j] = poincare_distance_cpu(q, db, curvature);
        }
    }

    distances
}

// ============================================================================
// TESTS - REAL DATA ONLY, NO MOCKS (per constitution REQ-KG-TEST)
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    // ========== Configuration Tests ==========

    #[test]
    fn test_config_default() {
        let config = PoincareCudaConfig::default();
        assert_eq!(config.dim, 64);
        assert!((config.curvature - (-1.0)).abs() < 1e-6);
    }

    #[test]
    fn test_config_with_curvature_valid() {
        let config = PoincareCudaConfig::with_curvature(-0.5).unwrap();
        assert!((config.curvature - (-0.5)).abs() < 1e-6);
        assert_eq!(config.dim, 64);
    }

    #[test]
    fn test_config_with_curvature_invalid_positive() {
        let result = PoincareCudaConfig::with_curvature(0.5);
        assert!(result.is_err());
        let err = result.unwrap_err();
        assert!(matches!(err, CudaError::InvalidConfig(_)));
    }

    #[test]
    fn test_config_with_curvature_invalid_zero() {
        let result = PoincareCudaConfig::with_curvature(0.0);
        assert!(result.is_err());
    }

    #[test]
    fn test_config_with_curvature_invalid_nan() {
        let result = PoincareCudaConfig::with_curvature(f32::NAN);
        assert!(result.is_err());
    }

    #[test]
    fn test_config_validate_default() {
        let config = PoincareCudaConfig::default();
        assert!(config.validate().is_ok());
    }

    #[test]
    fn test_config_validate_wrong_dimension() {
        let bad_config = PoincareCudaConfig {
            dim: 128,
            curvature: -1.0,
        };
        let result = bad_config.validate();
        assert!(result.is_err());
        let err_msg = result.unwrap_err().to_string();
        assert!(err_msg.contains("64"));
    }

    #[test]
    fn test_config_abs_curvature() {
        let config = PoincareCudaConfig::with_curvature(-2.5).unwrap();
        assert!((config.abs_curvature() - 2.5).abs() < 1e-6);
    }

    #[test]
    fn test_constants() {
        assert_eq!(POINCARE_DIM, 64);
        assert!((DEFAULT_CURVATURE - (-1.0)).abs() < 1e-6);
        assert!(POINCARE_EPS > 0.0);
        assert!(POINCARE_EPS < 1e-5);
    }

    // ========== CPU Reference Implementation Tests ==========

    #[test]
    fn test_cpu_distance_same_point_is_zero() {
        let point = [0.1f32; 64];
        let dist = poincare_distance_cpu(&point, &point, -1.0);
        assert!(dist.abs() < 1e-5, "Distance to self should be ~0, got {}", dist);
    }

    #[test]
    fn test_cpu_distance_origin_to_origin() {
        let origin = [0.0f32; 64];
        let dist = poincare_distance_cpu(&origin, &origin, -1.0);
        assert!(dist.abs() < 1e-6);
    }

    #[test]
    fn test_cpu_distance_is_symmetric() {
        let mut x = [0.0f32; 64];
        let mut y = [0.0f32; 64];
        x[0] = 0.3;
        y[0] = 0.6;

        let d1 = poincare_distance_cpu(&x, &y, -1.0);
        let d2 = poincare_distance_cpu(&y, &x, -1.0);

        assert!(
            (d1 - d2).abs() < 1e-5,
            "Distance should be symmetric: {} vs {}",
            d1,
            d2
        );
    }

    #[test]
    fn test_cpu_distance_is_nonnegative() {
        let mut x = [0.0f32; 64];
        let mut y = [0.0f32; 64];
        x[0] = 0.3;
        y[0] = -0.5;

        let dist = poincare_distance_cpu(&x, &y, -1.0);
        assert!(dist >= 0.0, "Distance should be non-negative");
    }

    #[test]
    fn test_cpu_distance_from_origin() {
        let origin = [0.0f32; 64];
        let mut point = [0.0f32; 64];
        point[0] = 0.5;

        let dist = poincare_distance_cpu(&origin, &point, -1.0);

        // For c=-1, using direct formula from origin:
        // d(0, p) = 2 * arctanh(sqrt(||p||² / (1 - ||p||²)))
        // With ||p||² = 0.25: arg = sqrt(0.25 / 0.75) = sqrt(1/3)
        // d(0, p) = 2 * arctanh(sqrt(1/3)) ≈ 1.317
        let norm_sq = 0.25_f32;
        let arg = (norm_sq / (1.0 - norm_sq)).sqrt();
        let expected = 2.0 * arg.atanh();
        assert!(
            (dist - expected).abs() < 1e-4,
            "Expected {}, got {}",
            expected,
            dist
        );
    }

    #[test]
    fn test_cpu_distance_monotonic_from_origin() {
        let origin = [0.0f32; 64];

        let mut p1 = [0.0f32; 64];
        p1[0] = 0.1;
        let mut p2 = [0.0f32; 64];
        p2[0] = 0.5;
        let mut p3 = [0.0f32; 64];
        p3[0] = 0.9;

        let d1 = poincare_distance_cpu(&origin, &p1, -1.0);
        let d2 = poincare_distance_cpu(&origin, &p2, -1.0);
        let d3 = poincare_distance_cpu(&origin, &p3, -1.0);

        assert!(d1 < d2, "d(0, 0.1) < d(0, 0.5)");
        assert!(d2 < d3, "d(0, 0.5) < d(0, 0.9)");
    }

    #[test]
    fn test_cpu_distance_near_boundary_large() {
        let origin = [0.0f32; 64];
        let mut near_boundary = [0.0f32; 64];
        near_boundary[0] = 0.99;

        let dist = poincare_distance_cpu(&origin, &near_boundary, -1.0);

        // Near boundary, hyperbolic distance grows rapidly
        assert!(
            dist > 4.0,
            "Distance near boundary should be large, got {}",
            dist
        );
    }

    #[test]
    fn test_cpu_distance_custom_curvature() {
        let mut x = [0.0f32; 64];
        let mut y = [0.0f32; 64];
        x[0] = 0.3;
        y[0] = 0.6;

        let d1 = poincare_distance_cpu(&x, &y, -1.0);
        let d2 = poincare_distance_cpu(&x, &y, -0.5);

        // Different curvatures should give different distances
        assert!((d1 - d2).abs() > 0.01, "Curvature should affect distance");
    }

    #[test]
    fn test_cpu_batch_dimensions() {
        let n_queries = 10;
        let n_database = 20;

        let queries: Vec<f32> = (0..(n_queries * 64)).map(|i| (i as f32) * 0.001).collect();
        let database: Vec<f32> = (0..(n_database * 64)).map(|i| (i as f32) * 0.0005).collect();

        let distances = poincare_distance_batch_cpu(&queries, &database, n_queries, n_database, -1.0);

        assert_eq!(distances.len(), n_queries * n_database);
    }

    #[test]
    fn test_cpu_batch_matches_single() {
        let n_queries = 5;
        let n_database = 5;

        // Create random-ish points (deterministic)
        let queries: Vec<f32> = (0..(n_queries * 64))
            .map(|i| ((i * 17 + 3) % 100) as f32 * 0.005 - 0.25)
            .collect();
        let database: Vec<f32> = (0..(n_database * 64))
            .map(|i| ((i * 23 + 7) % 100) as f32 * 0.005 - 0.25)
            .collect();

        let batch_distances =
            poincare_distance_batch_cpu(&queries, &database, n_queries, n_database, -1.0);

        // Compare with individual computations
        for i in 0..n_queries {
            for j in 0..n_database {
                let q: &[f32; 64] = queries[i * 64..(i + 1) * 64].try_into().unwrap();
                let db: &[f32; 64] = database[j * 64..(j + 1) * 64].try_into().unwrap();
                let single_dist = poincare_distance_cpu(q, db, -1.0);
                let batch_dist = batch_distances[i * n_database + j];

                assert!(
                    (single_dist - batch_dist).abs() < 1e-5,
                    "Mismatch at [{}, {}]: {} vs {}",
                    i,
                    j,
                    single_dist,
                    batch_dist
                );
            }
        }
    }

    // ========== Edge Case Tests ==========

    #[test]
    fn test_cpu_edge_case_zero_vectors() {
        let zero = [0.0f32; 64];
        let dist = poincare_distance_cpu(&zero, &zero, -1.0);
        assert!(dist.abs() < 1e-6, "Zero to zero should be 0");
    }

    #[test]
    fn test_cpu_edge_case_boundary_points() {
        // Points very close to boundary (norm ≈ 0.99)
        let scale = 0.99 / (64.0_f32).sqrt();
        let boundary_point: [f32; 64] = [scale; 64];

        let dist = poincare_distance_cpu(&boundary_point, &boundary_point, -1.0);
        assert!(dist.abs() < 1e-4, "Same point distance should be ~0");
        assert!(dist.is_finite(), "Distance must be finite");
    }

    #[test]
    fn test_cpu_edge_case_opposite_points() {
        // Points on opposite sides of the ball
        let mut x = [0.0f32; 64];
        let mut y = [0.0f32; 64];
        x[0] = 0.9;
        y[0] = -0.9;

        let dist = poincare_distance_cpu(&x, &y, -1.0);
        assert!(dist > 3.0, "Opposite points should have large distance: {}", dist);
        assert!(dist.is_finite(), "Distance must be finite");
    }

    // ========== GPU Availability Test ==========

    #[test]
    fn test_is_gpu_available_returns_bool() {
        // This test just ensures the function doesn't crash
        let _available = is_poincare_gpu_available();
        // Result depends on hardware - we just check it returns a bool
    }

    #[test]
    fn test_kernel_info_format() {
        // Check kernel info returns expected format
        if let Some(info) = get_kernel_info() {
            assert!(info.block_dim_x > 0);
            assert!(info.block_dim_y > 0);
            assert_eq!(info.point_dim, 64);
            assert!(info.shared_mem_bytes > 0);
        }
        // If None, CUDA feature is disabled - that's okay
    }
}
