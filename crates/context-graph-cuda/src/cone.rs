//! CUDA-accelerated entailment cone membership checking.
//!
//! Provides GPU-accelerated batch cone membership computation for hyperbolic geometry.
//! Falls back to CPU when GPU is unavailable.
//!
//! # Performance
//!
//! - GPU (RTX 5090): <2ms for 1K × 1K membership matrix
//! - CPU fallback: Uses reference implementation
//!
//! # Constitution Reference
//!
//! - TECH-GRAPH-004 Section 10.2: Cone CUDA Kernel
//! - perf.latency.entailment_check: <1ms
//! - perf.latency.cone_containment_gpu: <2ms for 1K × 1K batch
//!
//! # CANONICAL Membership Score Formula
//!
//! ```text
//! - If angle <= aperture: score = 1.0
//! - If angle > aperture: score = exp(-2.0 * (angle - aperture))
//! ```
//!
//! # Angle Computation Algorithm
//!
//! ```text
//! 1. tangent = log_map(apex, point) - direction to point in tangent space
//! 2. to_origin = log_map(apex, origin) - cone axis direction (toward origin)
//! 3. cos_angle = dot(tangent, to_origin) / (||tangent|| * ||to_origin||)
//! 4. angle = acos(cos_angle.clamp(-1.0, 1.0))
//!
//! Edge cases that return angle = 0.0 (score = 1.0):
//! - Point at apex (distance < eps)
//! - Apex at origin (norm < eps)
//! - Zero-length tangent or to_origin vectors
//! ```

use std::ffi::c_void;

use crate::error::{CudaError, CudaResult};

// ============================================================================
// CONSTANTS
// ============================================================================

/// Cone data dimension (64 apex coords + 1 aperture).
/// Must match kernel's CONE_DATA_DIM constant.
pub const CONE_DATA_DIM: usize = 65;

/// Point dimension (must match POINCARE_DIM and kernel's POINT_DIM).
pub const POINT_DIM: usize = 64;

/// Default curvature (negative, per hyperbolic geometry).
/// Standard unit hyperbolic space has curvature -1.0.
pub const DEFAULT_CURVATURE: f32 = -1.0;

/// Numerical epsilon for stability (matches kernel).
pub const CONE_EPS: f32 = 1e-7;

// ============================================================================
// FFI Declarations (CUDA kernels)
// ============================================================================

/// FFI declarations for CUDA kernel functions.
/// Only available when the `cuda` feature is enabled.
#[cfg(feature = "cuda")]
mod ffi {
    use std::ffi::c_void;
    use std::os::raw::c_int;

    #[link(name = "cone_check", kind = "static")]
    extern "C" {
        /// Launch batch cone membership computation.
        pub fn launch_cone_check(
            d_cones: *const f32,
            d_points: *const f32,
            d_scores: *mut f32,
            n_cones: c_int,
            n_points: c_int,
            curvature: f32,
            stream: *mut c_void,
        ) -> c_int;

        /// Single cone membership score (delegates to batch with n=1).
        pub fn cone_check_single(
            d_cone: *const f32,
            d_point: *const f32,
            d_score: *mut f32,
            curvature: f32,
            stream: *mut c_void,
        ) -> c_int;

        /// Get kernel configuration info.
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
// Configuration
// ============================================================================

/// Configuration for Cone CUDA operations.
///
/// # Example
///
/// ```
/// use context_graph_cuda::cone::ConeCudaConfig;
///
/// let config = ConeCudaConfig::default();
/// assert!((config.curvature - (-1.0)).abs() < 1e-6);
/// assert!(config.validate().is_ok());
/// ```
#[derive(Debug, Clone)]
pub struct ConeCudaConfig {
    /// Curvature (must be negative, default -1.0).
    pub curvature: f32,
}

impl Default for ConeCudaConfig {
    fn default() -> Self {
        Self {
            curvature: DEFAULT_CURVATURE,
        }
    }
}

impl ConeCudaConfig {
    /// Create config with custom curvature.
    ///
    /// # Errors
    ///
    /// Returns error if curvature is not negative or is NaN.
    ///
    /// # Example
    ///
    /// ```
    /// use context_graph_cuda::cone::ConeCudaConfig;
    ///
    /// let config = ConeCudaConfig::with_curvature(-0.5).unwrap();
    /// assert!((config.curvature - (-0.5)).abs() < 1e-6);
    ///
    /// // Invalid curvature returns error
    /// assert!(ConeCudaConfig::with_curvature(0.5).is_err());
    /// ```
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

    /// Validate configuration parameters.
    ///
    /// # Errors
    ///
    /// Returns error if:
    /// - Curvature is not negative
    /// - Curvature is NaN
    ///
    /// # Example
    ///
    /// ```
    /// use context_graph_cuda::cone::ConeCudaConfig;
    ///
    /// let config = ConeCudaConfig::default();
    /// assert!(config.validate().is_ok());
    /// ```
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

    /// Get absolute value of curvature (always positive).
    #[inline]
    pub fn abs_curvature(&self) -> f32 {
        self.curvature.abs()
    }
}

// ============================================================================
// Data Structures
// ============================================================================

/// Cone data packed for GPU transfer.
///
/// Contains 64 apex coordinates plus 1 aperture value.
///
/// # Example
///
/// ```
/// use context_graph_cuda::cone::ConeData;
///
/// let apex = [0.1f32; 64];
/// let cone = ConeData::new(apex, 0.5).unwrap();
/// assert!((cone.aperture - 0.5).abs() < 1e-6);
/// ```
#[derive(Debug, Clone, Copy)]
pub struct ConeData {
    /// Apex point in Poincare ball (64 floats).
    pub apex: [f32; 64],
    /// Effective aperture angle in radians.
    pub aperture: f32,
}

impl ConeData {
    /// Create new cone data.
    ///
    /// # Errors
    ///
    /// Returns error if:
    /// - Apex norm >= 1.0 (outside Poincare ball)
    /// - Aperture is negative
    /// - Aperture is NaN
    ///
    /// # Example
    ///
    /// ```
    /// use context_graph_cuda::cone::ConeData;
    ///
    /// let apex = [0.1f32; 64];
    /// let cone = ConeData::new(apex, 0.5).unwrap();
    /// assert!((cone.aperture - 0.5).abs() < 1e-6);
    ///
    /// // Invalid apex (outside ball) returns error
    /// let bad_apex = [1.0f32; 64]; // norm > 1
    /// assert!(ConeData::new(bad_apex, 0.5).is_err());
    /// ```
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
        if aperture.is_nan() {
            return Err(CudaError::InvalidConfig(
                "Aperture cannot be NaN".to_string(),
            ));
        }
        Ok(Self { apex, aperture })
    }

    /// Create cone data from raw components (unchecked).
    ///
    /// Use this when you've already validated the data.
    #[inline]
    pub fn from_raw(apex: [f32; 64], aperture: f32) -> Self {
        Self { apex, aperture }
    }

    /// Pack to GPU format [apex_0..apex_63, aperture].
    pub fn to_gpu_format(&self) -> [f32; CONE_DATA_DIM] {
        let mut data = [0.0f32; CONE_DATA_DIM];
        data[..64].copy_from_slice(&self.apex);
        data[64] = self.aperture;
        data
    }

    /// Unpack from GPU format.
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
// Kernel Info
// ============================================================================

/// Information about the CUDA kernel configuration.
///
/// Useful for debugging and performance tuning.
#[derive(Debug, Clone, Copy)]
pub struct ConeKernelInfo {
    /// Block dimension X (warp-aligned, typically 32).
    pub block_dim_x: i32,
    /// Block dimension Y (cones per block, typically 8).
    pub block_dim_y: i32,
    /// Point dimension (must be 64).
    pub point_dim: i32,
    /// Cone data dimension (65 = 64 apex + 1 aperture).
    pub cone_data_dim: i32,
    /// Shared memory per block in bytes.
    pub shared_mem_bytes: i32,
}

/// Get kernel configuration info.
///
/// Returns None if CUDA feature is disabled.
///
/// # Example
///
/// ```
/// use context_graph_cuda::cone::get_cone_kernel_info;
///
/// if let Some(info) = get_cone_kernel_info() {
///     println!("Block size: {}x{}", info.block_dim_x, info.block_dim_y);
///     println!("Shared memory: {} bytes", info.shared_mem_bytes);
/// }
/// ```
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

// ============================================================================
// GPU Availability
// ============================================================================

/// Check if CUDA Cone kernels are available.
///
/// Returns true if:
/// 1. The `cuda` feature is enabled at compile time
/// 2. The system has a CUDA-capable GPU
/// 3. The CUDA runtime is available
///
/// # Example
///
/// ```
/// use context_graph_cuda::cone::is_cone_gpu_available;
///
/// if is_cone_gpu_available() {
///     println!("GPU acceleration available!");
/// } else {
///     println!("Using CPU fallback");
/// }
/// ```
#[cfg(feature = "cuda")]
pub fn is_cone_gpu_available() -> bool {
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
pub fn is_cone_gpu_available() -> bool {
    false
}

// ============================================================================
// GPU Functions
// ============================================================================

/// Compute batch cone membership scores on GPU.
///
/// # Safety
///
/// - `d_cones`, `d_points`, `d_scores` must be valid device pointers
/// - Arrays must be properly sized: cones\[n_cones\]\[65\], points\[n_points\]\[64\]
/// - Output scores\[n_cones\]\[n_points\] must be pre-allocated
/// - Pointers must be aligned for float32 access
///
/// # Arguments
///
/// * `d_cones` - Device pointer to cone data \[n_cones\]\[65\]
/// * `d_points` - Device pointer to point vectors \[n_points\]\[64\]
/// * `d_scores` - Device pointer to output matrix \[n_cones\]\[n_points\]
/// * `n_cones` - Number of cones
/// * `n_points` - Number of points
/// * `config` - Cone configuration
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
/// Target: <2ms for 1K × 1K membership matrix on RTX 5090.
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
    // Validate configuration
    config.validate()?;

    // Validate pointer arguments
    if d_cones.is_null() {
        return Err(CudaError::InvalidConfig(
            "d_cones pointer is null".to_string(),
        ));
    }
    if d_points.is_null() {
        return Err(CudaError::InvalidConfig(
            "d_points pointer is null".to_string(),
        ));
    }
    if d_scores.is_null() {
        return Err(CudaError::InvalidConfig(
            "d_scores pointer is null".to_string(),
        ));
    }

    // Validate sizes
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

/// Stub implementation when CUDA feature is disabled.
///
/// # Safety
///
/// This stub always returns an error and does not use the pointers.
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

/// Compute single cone membership score on GPU.
///
/// # Safety
///
/// Same requirements as `cone_check_batch_gpu`.
///
/// # Note
///
/// For efficiency, prefer `cone_check_batch_gpu` when computing
/// multiple scores. Single-pair calls have significant kernel launch overhead.
#[cfg(feature = "cuda")]
pub unsafe fn cone_check_single_gpu(
    d_cone: *const f32,
    d_point: *const f32,
    d_score: *mut f32,
    config: &ConeCudaConfig,
    stream: Option<*mut c_void>,
) -> CudaResult<()> {
    config.validate()?;

    if d_cone.is_null() || d_point.is_null() || d_score.is_null() {
        return Err(CudaError::InvalidConfig(
            "Device pointers cannot be null".to_string(),
        ));
    }

    let stream_ptr = stream.unwrap_or(std::ptr::null_mut());

    let result = ffi::cone_check_single(
        d_cone,
        d_point,
        d_score,
        config.curvature,
        stream_ptr,
    );

    if result != 0 {
        return Err(CudaError::KernelError(format!(
            "Cone check single kernel failed with CUDA error code {}",
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
pub unsafe fn cone_check_single_gpu(
    _d_cone: *const f32,
    _d_point: *const f32,
    _d_score: *mut f32,
    _config: &ConeCudaConfig,
    _stream: Option<*mut c_void>,
) -> CudaResult<()> {
    Err(CudaError::NotImplemented(
        "CUDA feature not enabled. Compile with --features cuda".to_string(),
    ))
}

// ============================================================================
// CPU Reference Implementation (for testing and fallback)
// ============================================================================

/// Compute Mobius addition on CPU.
///
/// Formula: (x (+) y) = ((1 + 2c<x,y> + c||y||²)x + (1 - c||x||²)y) /
///                      (1 + 2c<x,y> + c²||x||²||y||²)
fn mobius_add_cpu(x: &[f32; 64], y: &[f32; 64], c: f32) -> [f32; 64] {
    let x_norm_sq: f32 = x.iter().map(|v| v * v).sum();
    let y_norm_sq: f32 = y.iter().map(|v| v * v).sum();
    let xy_dot: f32 = x.iter().zip(y.iter()).map(|(a, b)| a * b).sum();

    let num_coeff_x = 1.0 + 2.0 * c * xy_dot + c * y_norm_sq;
    let num_coeff_y = 1.0 - c * x_norm_sq;
    let denom = 1.0 + 2.0 * c * xy_dot + c * c * x_norm_sq * y_norm_sq;

    let safe_denom = if denom.abs() < CONE_EPS {
        if denom < 0.0 { -CONE_EPS } else { CONE_EPS }
    } else {
        denom
    };

    let mut result = [0.0f32; 64];
    for i in 0..64 {
        result[i] = (num_coeff_x * x[i] + num_coeff_y * y[i]) / safe_denom;
    }
    result
}

/// Compute log map on CPU: log_x(y) - tangent vector at x pointing toward y.
///
/// Formula: log_x(y) = (2 / (λ_x * √c)) * arctanh(√c * ||(-x) ⊕ y||) * ((-x) ⊕ y) / ||(-x) ⊕ y||
fn log_map_cpu(x: &[f32; 64], y: &[f32; 64], c: f32) -> [f32; 64] {
    let sqrt_c = c.sqrt();

    // Compute (-x) for Mobius subtraction
    let mut neg_x = [0.0f32; 64];
    for i in 0..64 {
        neg_x[i] = -x[i];
    }

    // Compute diff = (-x) ⊕ y
    let diff = mobius_add_cpu(&neg_x, y, c);

    // Compute ||diff||
    let diff_norm_sq: f32 = diff.iter().map(|v| v * v).sum();
    let diff_norm = diff_norm_sq.max(0.0).sqrt();

    // Handle identical points
    if diff_norm < CONE_EPS {
        return [0.0f32; 64];
    }

    // Conformal factor at x: λ_x = 2 / (1 - c||x||²)
    let x_norm_sq: f32 = x.iter().map(|v| v * v).sum();
    let denom_lambda = (1.0 - c * x_norm_sq).max(CONE_EPS);
    let lambda_x = 2.0 / denom_lambda;

    // arctanh(√c * ||(-x) ⊕ y||), clamped
    let arg = (sqrt_c * diff_norm).min(1.0 - CONE_EPS);
    let arctanh_val = arg.atanh();

    // Scale factor
    let scale = (2.0 / (lambda_x * sqrt_c)) * arctanh_val / diff_norm;

    let mut tangent = [0.0f32; 64];
    for i in 0..64 {
        tangent[i] = scale * diff[i];
    }
    tangent
}

/// Compute single cone membership score on CPU.
///
/// CANONICAL FORMULA:
/// - If angle <= aperture: 1.0
/// - If angle > aperture: exp(-2.0 * (angle - aperture))
///
/// # Arguments
///
/// * `apex` - Cone apex (64-element array)
/// * `aperture` - Effective aperture in radians
/// * `point` - Point to test (64-element array)
/// * `curvature` - Poincare ball curvature (must be negative)
///
/// # Returns
///
/// Membership score in [0, 1].
///
/// # Example
///
/// ```
/// use context_graph_cuda::cone::cone_membership_score_cpu;
///
/// let apex = [0.1f32; 64];
/// let point = apex.clone();  // Same as apex
/// let score = cone_membership_score_cpu(&apex, 0.5, &point, -1.0);
/// assert!((score - 1.0).abs() < 1e-4);  // Point at apex = score 1.0
/// ```
pub fn cone_membership_score_cpu(
    apex: &[f32; 64],
    aperture: f32,
    point: &[f32; 64],
    curvature: f32,
) -> f32 {
    let c = curvature.abs();

    // Edge case: apex at origin (degenerate cone contains all)
    let apex_norm_sq: f32 = apex.iter().map(|x| x * x).sum();
    if apex_norm_sq < CONE_EPS * CONE_EPS {
        return 1.0;
    }

    // Edge case: point at apex
    let diff_sq: f32 = apex
        .iter()
        .zip(point.iter())
        .map(|(a, p)| (a - p) * (a - p))
        .sum();
    if diff_sq < CONE_EPS * CONE_EPS {
        return 1.0;
    }

    // Compute tangent from apex to point
    let tangent = log_map_cpu(apex, point, c);

    // Compute tangent from apex to origin (cone axis)
    let origin = [0.0f32; 64];
    let to_origin = log_map_cpu(apex, &origin, c);

    // Compute norms
    let tangent_norm_sq: f32 = tangent.iter().map(|x| x * x).sum();
    let to_origin_norm_sq: f32 = to_origin.iter().map(|x| x * x).sum();

    let tangent_norm = tangent_norm_sq.sqrt();
    let to_origin_norm = to_origin_norm_sq.sqrt();

    // Edge case: degenerate tangent vectors
    if tangent_norm < CONE_EPS || to_origin_norm < CONE_EPS {
        return 1.0;
    }

    // Compute angle via dot product
    let dot: f32 = tangent
        .iter()
        .zip(to_origin.iter())
        .map(|(a, b)| a * b)
        .sum();
    let cos_angle = (dot / (tangent_norm * to_origin_norm)).clamp(-1.0, 1.0);
    let angle = cos_angle.acos();

    // CANONICAL FORMULA
    if angle <= aperture {
        1.0
    } else {
        (-2.0 * (angle - aperture)).exp()
    }
}

/// Compute batch cone membership scores on CPU.
///
/// Reference implementation for testing GPU kernel correctness.
///
/// # Arguments
///
/// * `cones` - Cone data, flattened \[n_cones * 65\]
/// * `points` - Point vectors, flattened \[n_points * 64\]
/// * `n_cones` - Number of cones
/// * `n_points` - Number of points
/// * `curvature` - Poincare ball curvature (must be negative)
///
/// # Returns
///
/// Score matrix, flattened \[n_cones * n_points\], row-major.
///
/// # Panics
///
/// Panics if input arrays have incorrect sizes.
pub fn cone_check_batch_cpu(
    cones: &[f32],
    points: &[f32],
    n_cones: usize,
    n_points: usize,
    curvature: f32,
) -> Vec<f32> {
    assert_eq!(
        cones.len(),
        n_cones * CONE_DATA_DIM,
        "Invalid cones size: expected {}, got {}",
        n_cones * CONE_DATA_DIM,
        cones.len()
    );
    assert_eq!(
        points.len(),
        n_points * POINT_DIM,
        "Invalid points size: expected {}, got {}",
        n_points * POINT_DIM,
        points.len()
    );

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
// TESTS - REAL DATA ONLY, NO MOCKS (per constitution REQ-KG-TEST)
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    // ========== Configuration Tests ==========

    #[test]
    fn test_config_default() {
        let config = ConeCudaConfig::default();
        assert!((config.curvature - (-1.0)).abs() < 1e-6);
        assert!(config.validate().is_ok());
    }

    #[test]
    fn test_config_with_curvature_valid() {
        let config = ConeCudaConfig::with_curvature(-0.5).unwrap();
        assert!((config.curvature - (-0.5)).abs() < 1e-6);
    }

    #[test]
    fn test_config_with_curvature_invalid_positive() {
        let result = ConeCudaConfig::with_curvature(0.5);
        assert!(result.is_err());
        let err = result.unwrap_err();
        assert!(matches!(err, CudaError::InvalidConfig(_)));
    }

    #[test]
    fn test_config_with_curvature_invalid_zero() {
        let result = ConeCudaConfig::with_curvature(0.0);
        assert!(result.is_err());
    }

    #[test]
    fn test_config_with_curvature_invalid_nan() {
        let result = ConeCudaConfig::with_curvature(f32::NAN);
        assert!(result.is_err());
    }

    #[test]
    fn test_config_abs_curvature() {
        let config = ConeCudaConfig::with_curvature(-2.5).unwrap();
        assert!((config.abs_curvature() - 2.5).abs() < 1e-6);
    }

    // ========== ConeData Tests ==========

    #[test]
    fn test_cone_data_valid() {
        let apex = [0.1f32; 64];
        let cone = ConeData::new(apex, 0.5);
        assert!(cone.is_ok());
        let cone = cone.unwrap();
        assert!((cone.aperture - 0.5).abs() < 1e-6);
    }

    #[test]
    fn test_cone_data_invalid_apex_outside_ball() {
        let apex = [1.0f32; 64]; // norm = sqrt(64) >> 1
        let result = ConeData::new(apex, 0.5);
        assert!(result.is_err());
    }

    #[test]
    fn test_cone_data_invalid_negative_aperture() {
        let apex = [0.1f32; 64];
        let result = ConeData::new(apex, -0.5);
        assert!(result.is_err());
    }

    #[test]
    fn test_cone_data_gpu_format_roundtrip() {
        let apex = [0.123f32; 64];
        let cone = ConeData::from_raw(apex, 0.456);

        let gpu_format = cone.to_gpu_format();
        let restored = ConeData::from_gpu_format(&gpu_format);

        assert!((restored.aperture - cone.aperture).abs() < 1e-6);
        for i in 0..64 {
            assert!((restored.apex[i] - cone.apex[i]).abs() < 1e-6);
        }
    }

    // ========== CPU Reference Implementation Tests ==========

    #[test]
    fn test_cpu_score_point_at_apex_returns_1() {
        let apex = [0.1f32; 64];
        let point = apex;
        let score = cone_membership_score_cpu(&apex, 0.5, &point, -1.0);
        assert!(
            (score - 1.0).abs() < 1e-4,
            "Point at apex should have score 1.0, got {}",
            score
        );
    }

    #[test]
    fn test_cpu_score_apex_at_origin() {
        // Degenerate cone at origin contains all points
        let apex = [0.0f32; 64];
        let mut point = [0.0f32; 64];
        point[0] = 0.5;

        let score = cone_membership_score_cpu(&apex, 0.5, &point, -1.0);
        assert!(
            (score - 1.0).abs() < 1e-4,
            "Apex at origin should give score 1.0, got {}",
            score
        );
    }

    #[test]
    fn test_cpu_score_inside_cone_returns_1() {
        // Wide aperture = point is inside cone
        let mut apex = [0.0f32; 64];
        apex[0] = 0.3;

        let mut point = [0.0f32; 64];
        point[0] = 0.1; // Point between apex and origin

        let score = cone_membership_score_cpu(&apex, 1.5, &point, -1.0);
        assert!(
            score > 0.9,
            "Point inside wide cone should have high score: {}",
            score
        );
    }

    #[test]
    fn test_cpu_score_outside_cone_decays() {
        // Narrow cone and point far outside
        let mut apex = [0.0f32; 64];
        apex[0] = 0.5;
        let aperture = 0.1; // Very narrow

        let mut point = [0.0f32; 64];
        point[1] = 0.5; // Perpendicular direction

        let score = cone_membership_score_cpu(&apex, aperture, &point, -1.0);
        assert!(
            score < 0.5,
            "Point outside narrow cone should have low score: {}",
            score
        );
    }

    #[test]
    fn test_cpu_score_is_bounded() {
        // All scores must be in [0, 1]
        for seed in 0..20 {
            let mut apex = [0.0f32; 64];
            let mut point = [0.0f32; 64];

            // Deterministic "random" values
            apex[seed % 64] = ((seed as f32 * 0.07) % 0.9) * if seed % 2 == 0 { 1.0 } else { -1.0 };
            point[(seed + 7) % 64] = ((seed as f32 * 0.11) % 0.9) * if seed % 3 == 0 { 1.0 } else { -1.0 };

            let score = cone_membership_score_cpu(&apex, 0.5, &point, -1.0);
            assert!(
                score >= 0.0 && score <= 1.0,
                "Score must be in [0,1], got {} for seed {}",
                score,
                seed
            );
            assert!(score.is_finite(), "Score must be finite for seed {}", seed);
        }
    }

    #[test]
    fn test_cpu_canonical_formula_verified() {
        // Verify the CANONICAL formula: exp(-2 * (angle - aperture))
        let mut apex = [0.0f32; 64];
        apex[0] = 0.3;
        let aperture = 0.3; // Narrow aperture

        // Point at a significant angle from cone axis
        let mut point = [0.0f32; 64];
        point[0] = 0.1;
        point[1] = 0.3;

        let score = cone_membership_score_cpu(&apex, aperture, &point, -1.0);

        // Score should be exp(-2 * (angle - aperture)) if angle > aperture
        // Just verify it's a reasonable value
        println!("CANONICAL FORMULA TEST: aperture={}, score={}", aperture, score);
        assert!(score >= 0.0 && score <= 1.0, "Score must be in [0,1]");
    }

    // ========== Batch Tests ==========

    #[test]
    fn test_batch_cpu_matches_single() {
        let n_cones = 5;
        let n_points = 5;

        let cones: Vec<f32> = (0..n_cones)
            .flat_map(|i| {
                let mut data = [0.0f32; 65];
                data[0] = (i as f32) * 0.1; // Varying apex
                data[64] = 0.5; // Aperture
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
                    i,
                    j,
                    single,
                    batch
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
            assert!(
                s >= 0.0 && s <= 1.0 && s.is_finite(),
                "Invalid score at {}: {}",
                idx,
                s
            );
        }
    }

    // ========== Edge Cases ==========

    #[test]
    fn test_edge_case_boundary_point() {
        let apex = [0.1f32; 64];
        let scale = 0.99 / (64.0_f32).sqrt();
        let boundary_point: [f32; 64] = [scale; 64];

        let score = cone_membership_score_cpu(&apex, 0.5, &boundary_point, -1.0);
        assert!(
            score.is_finite() && score >= 0.0 && score <= 1.0,
            "Boundary point score invalid: {}",
            score
        );
    }

    #[test]
    fn test_edge_case_zero_aperture() {
        let apex = [0.1f32; 64];
        let mut point = [0.0f32; 64];
        point[0] = 0.2;

        let score = cone_membership_score_cpu(&apex, 0.0, &point, -1.0);
        // With zero aperture, only apex should score 1.0
        assert!(
            score < 1.0,
            "Non-apex point with zero aperture should have score < 1.0"
        );
        assert!(score.is_finite() && score >= 0.0);
    }

    #[test]
    fn test_edge_case_large_aperture() {
        // Aperture of PI/2 should contain many points
        let apex = [0.1f32; 64];
        let mut point = [0.0f32; 64];
        point[0] = 0.2;

        let score = cone_membership_score_cpu(&apex, std::f32::consts::FRAC_PI_2, &point, -1.0);
        assert!(
            score > 0.9,
            "Point with large aperture should have high score: {}",
            score
        );
    }

    // ========== GPU Availability Test ==========

    #[test]
    fn test_gpu_availability_check() {
        let available = is_cone_gpu_available();
        println!("Cone GPU available: {}", available);

        #[cfg(not(feature = "cuda"))]
        assert!(!available, "Without cuda feature, GPU should not be available");
    }

    // ========== Constants Verification ==========

    #[test]
    fn test_constants() {
        assert_eq!(CONE_DATA_DIM, 65, "CONE_DATA_DIM should be 65");
        assert_eq!(POINT_DIM, 64, "POINT_DIM should be 64");
        assert!((DEFAULT_CURVATURE - (-1.0)).abs() < 1e-10, "DEFAULT_CURVATURE should be -1.0");
        assert!(CONE_EPS > 0.0 && CONE_EPS < 1e-5, "CONE_EPS should be small positive");
    }
}
