// CUDA kernel for batch entailment cone membership computation
// Target: RTX 5090 (Compute Capability 12.0, CUDA 13.1)
// Performance: <2ms for 1K x 1K membership matrix
//
// Constitution Reference:
// - TECH-GRAPH-004 Section 10.2: Cone CUDA Kernel
// - perf.latency.entailment_check: <1ms
// - perf.latency.cone_containment_gpu: <2ms for 1K x 1K batch
//
// Mathematics:
// CANONICAL membership score formula:
// - If angle <= aperture: score = 1.0
// - If angle > aperture: score = exp(-2.0 * (angle - aperture))
//
// Angle Computation Algorithm:
// 1. tangent = log_map(apex, point) - direction to point in tangent space
// 2. to_origin = log_map(apex, origin) - cone axis direction (toward origin)
// 3. cos_angle = dot(tangent, to_origin) / (||tangent|| * ||to_origin||)
// 4. angle = acos(cos_angle.clamp(-1.0, 1.0))

#include <cuda_runtime.h>
#include <math.h>

// ============================================================================
// KERNEL CONFIGURATION - Tuned for RTX 5090 (170 SMs, 21760 CUDA cores)
// Block: 32 x 8 = 256 threads (optimal for Blackwell)
// Shared memory: 8 cones × 65 floats × 4 bytes = 2080 bytes per block
// ============================================================================
constexpr int BLOCK_DIM_X = 32;       // Warp size for coalesced access (points)
constexpr int BLOCK_DIM_Y = 8;        // Cones per block
constexpr int POINT_DIM = 64;         // Poincare ball dimension (fixed)
constexpr int CONE_DATA_DIM = 65;     // 64 apex coords + 1 aperture
constexpr int CONES_PER_BLOCK = 8;    // Matches BLOCK_DIM_Y

// Numerical stability constants
constexpr float EPS = 1e-7f;          // Prevent division by zero
constexpr float ARCTANH_CLAMP = 1.0f - 1e-7f;  // arctanh domain: (-1, 1)

// ============================================================================
// DEVICE HELPER FUNCTIONS
// ============================================================================

/**
 * Compute squared norm of a vector in registers.
 */
__device__ __forceinline__ float compute_norm_sq(const float* v) {
    float norm_sq = 0.0f;
    #pragma unroll 8
    for (int i = 0; i < POINT_DIM; i++) {
        norm_sq += v[i] * v[i];
    }
    return norm_sq;
}

/**
 * Compute dot product of two vectors in registers.
 */
__device__ __forceinline__ float compute_dot(const float* a, const float* b) {
    float dot = 0.0f;
    #pragma unroll 8
    for (int i = 0; i < POINT_DIM; i++) {
        dot += a[i] * b[i];
    }
    return dot;
}

/**
 * Compute Mobius addition: x (+) y in Poincare ball
 *
 * Formula: (x (+) y) = ((1 + 2c<x,y> + c||y||²)x + (1 - c||x||²)y) /
 *                      (1 + 2c<x,y> + c²||x||²||y||²)
 *
 * @param x       First point [POINT_DIM]
 * @param y       Second point [POINT_DIM]
 * @param result  Output point [POINT_DIM]
 * @param c       Curvature magnitude (always positive)
 */
__device__ void mobius_add(
    const float* x,
    const float* y,
    float* result,
    float c
) {
    // Compute norms
    float x_norm_sq = compute_norm_sq(x);
    float y_norm_sq = compute_norm_sq(y);

    // Inner product <x, y>
    float xy_dot = compute_dot(x, y);

    // Numerator coefficients
    float num_coeff_x = 1.0f + 2.0f * c * xy_dot + c * y_norm_sq;
    float num_coeff_y = 1.0f - c * x_norm_sq;

    // Denominator
    float denom = 1.0f + 2.0f * c * xy_dot + c * c * x_norm_sq * y_norm_sq;

    // Avoid division by zero
    float safe_denom = fmaxf(fabsf(denom), EPS);
    if (denom < 0.0f) safe_denom = -safe_denom;

    // Compute result
    #pragma unroll 8
    for (int i = 0; i < POINT_DIM; i++) {
        result[i] = (num_coeff_x * x[i] + num_coeff_y * y[i]) / safe_denom;
    }
}

/**
 * Compute log map: log_x(y) - tangent vector at x pointing toward y
 *
 * Formula: log_x(y) = (2 / (λ_x * √c)) * arctanh(√c * ||(-x) ⊕ y||) * ((-x) ⊕ y) / ||(-x) ⊕ y||
 * where λ_x = 2 / (1 - c||x||²) is the conformal factor.
 *
 * This is the key operation for computing the angle between point and cone axis.
 *
 * @param x       Base point [POINT_DIM]
 * @param y       Target point [POINT_DIM]
 * @param tangent Output tangent vector [POINT_DIM]
 * @param c       Curvature magnitude (always positive)
 */
__device__ void log_map(
    const float* x,
    const float* y,
    float* tangent,
    float c
) {
    float sqrt_c = sqrtf(c);

    // Compute (-x) for Mobius subtraction
    float neg_x[POINT_DIM];
    #pragma unroll 8
    for (int i = 0; i < POINT_DIM; i++) {
        neg_x[i] = -x[i];
    }

    // Compute diff = (-x) ⊕ y
    float diff[POINT_DIM];
    mobius_add(neg_x, y, diff, c);

    // Compute ||diff||
    float diff_norm_sq = compute_norm_sq(diff);
    float diff_norm = sqrtf(fmaxf(diff_norm_sq, 0.0f));

    // Handle identical points (return zero tangent)
    if (diff_norm < EPS) {
        #pragma unroll 8
        for (int i = 0; i < POINT_DIM; i++) {
            tangent[i] = 0.0f;
        }
        return;
    }

    // Conformal factor at x: λ_x = 2 / (1 - c||x||²)
    float x_norm_sq = compute_norm_sq(x);
    float denom_lambda = fmaxf(1.0f - c * x_norm_sq, EPS);
    float lambda_x = 2.0f / denom_lambda;

    // arctanh(√c * ||(-x) ⊕ y||), clamped to avoid NaN
    float arg = fminf(sqrt_c * diff_norm, ARCTANH_CLAMP);
    float arctanh_val = atanhf(arg);

    // Scale: (2 / (λ_x * √c)) * arctanh(...) / ||diff||
    float scale = (2.0f / (lambda_x * sqrt_c)) * arctanh_val / diff_norm;

    // Compute tangent vector
    #pragma unroll 8
    for (int i = 0; i < POINT_DIM; i++) {
        tangent[i] = scale * diff[i];
    }
}

/**
 * Compute cone membership score.
 *
 * CANONICAL FORMULA (DO NOT MODIFY):
 * 1. Compute tangent from apex to point
 * 2. Compute tangent from apex to origin (cone axis)
 * 3. Compute angle between tangents
 * 4. Apply score formula:
 *    - angle <= aperture: 1.0
 *    - angle > aperture: exp(-2.0 * (angle - aperture))
 *
 * Edge cases that return score = 1.0 (angle = 0):
 * - Point at apex (distance < eps)
 * - Apex at origin (norm < eps)
 * - Zero-length tangent or to_origin vectors
 *
 * @param apex     Cone apex point [POINT_DIM]
 * @param aperture Effective aperture angle in radians
 * @param point    Point to test [POINT_DIM]
 * @param c        Curvature magnitude (always positive)
 * @return         Membership score in [0, 1]
 */
__device__ float cone_membership_score(
    const float* apex,
    float aperture,
    const float* point,
    float c
) {
    // Edge case: apex at origin (degenerate cone contains all points)
    float apex_norm_sq = compute_norm_sq(apex);
    if (apex_norm_sq < EPS * EPS) {
        return 1.0f;
    }

    // Edge case: point at apex
    float diff_sq = 0.0f;
    #pragma unroll 8
    for (int i = 0; i < POINT_DIM; i++) {
        float d = point[i] - apex[i];
        diff_sq += d * d;
    }
    if (diff_sq < EPS * EPS) {
        return 1.0f;
    }

    // Compute tangent from apex to point
    float tangent[POINT_DIM];
    log_map(apex, point, tangent, c);

    // Compute tangent from apex to origin (cone axis direction)
    // Origin is all zeros
    float origin[POINT_DIM];
    #pragma unroll 8
    for (int i = 0; i < POINT_DIM; i++) {
        origin[i] = 0.0f;
    }

    float to_origin[POINT_DIM];
    log_map(apex, origin, to_origin, c);

    // Compute norms of tangent vectors
    float tangent_norm_sq = compute_norm_sq(tangent);
    float to_origin_norm_sq = compute_norm_sq(to_origin);

    float tangent_norm = sqrtf(fmaxf(tangent_norm_sq, 0.0f));
    float to_origin_norm = sqrtf(fmaxf(to_origin_norm_sq, 0.0f));

    // Edge case: degenerate tangent vectors
    if (tangent_norm < EPS || to_origin_norm < EPS) {
        return 1.0f;
    }

    // Compute angle via dot product
    float dot = compute_dot(tangent, to_origin);
    float cos_angle = dot / (tangent_norm * to_origin_norm);

    // Clamp to valid acos domain [-1, 1]
    cos_angle = fmaxf(-1.0f, fminf(1.0f, cos_angle));
    float angle = acosf(cos_angle);

    // CANONICAL FORMULA
    if (angle <= aperture) {
        return 1.0f;
    } else {
        return expf(-2.0f * (angle - aperture));
    }
}

// ============================================================================
// MAIN KERNEL
// ============================================================================

/**
 * Batch cone membership kernel.
 *
 * Grid: ((n_points + 31) / 32, (n_cones + 7) / 8)
 * Block: (32, 8) = 256 threads
 * Shared: 8 * 65 * 4 = 2080 bytes per block
 *
 * Memory access pattern:
 * - Cones: Cached in shared memory (reused by all threads in a row)
 * - Points: Coalesced global reads (adjacent threads read adjacent elements)
 * - Output: Row-major order, coalesced writes
 *
 * @param cones      Cone data [n_cones][65] - row-major, 65 = 64 apex coords + 1 aperture
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
) {
    // Shared memory for cone data (apex + aperture)
    // Cone caching provides ~32x reuse per thread block
    __shared__ float shared_cones[CONES_PER_BLOCK][CONE_DATA_DIM];

    const int tx = threadIdx.x;  // 0-31: point index within tile
    const int ty = threadIdx.y;  // 0-7: cone index within block
    const int bx = blockIdx.x;   // Point tile index
    const int by = blockIdx.y;   // Cone tile index

    const int point_idx = bx * BLOCK_DIM_X + tx;
    const int cone_idx = by * CONES_PER_BLOCK + ty;

    // Curvature magnitude (always positive for computations)
    const float c = fabsf(curvature);

    // =========================================================================
    // Phase 1: Load cone data into shared memory
    // =========================================================================
    // Each thread in a row collaboratively loads the cone data
    // Thread tx loads elements d, d+32, d+64, ... (strided for coalescing)
    if (cone_idx < n_cones) {
        for (int d = tx; d < CONE_DATA_DIM; d += BLOCK_DIM_X) {
            shared_cones[ty][d] = cones[cone_idx * CONE_DATA_DIM + d];
        }
    }
    __syncthreads();

    // =========================================================================
    // Phase 2: Compute membership score for this (cone, point) pair
    // =========================================================================
    if (cone_idx < n_cones && point_idx < n_points) {
        // Extract apex and aperture from shared memory
        float apex[POINT_DIM];
        #pragma unroll 8
        for (int i = 0; i < POINT_DIM; i++) {
            apex[i] = shared_cones[ty][i];
        }
        float aperture = shared_cones[ty][POINT_DIM];  // Last element is aperture

        // Load point from global memory into registers
        float point[POINT_DIM];
        #pragma unroll 8
        for (int i = 0; i < POINT_DIM; i++) {
            point[i] = points[point_idx * POINT_DIM + i];
        }

        // Compute membership score
        float score = cone_membership_score(apex, aperture, point, c);

        // Write result to global memory (row-major: [n_cones][n_points])
        scores[cone_idx * n_points + point_idx] = score;
    }
}

// ============================================================================
// HOST LAUNCHER
// ============================================================================

/**
 * Launch cone check kernel from host code.
 *
 * Automatically calculates optimal grid dimensions based on input sizes.
 * Validates curvature is negative (required for hyperbolic space).
 *
 * @param d_cones     Device pointer to cone data [n_cones][65]
 * @param d_points    Device pointer to point vectors [n_points][64]
 * @param d_scores    Device pointer to output scores [n_cones][n_points]
 * @param n_cones     Number of cones
 * @param n_points    Number of points
 * @param curvature   Poincare ball curvature (must be negative)
 * @param stream      CUDA stream (nullptr for default stream)
 * @return            0 (cudaSuccess) on success, CUDA error code on failure
 */
extern "C" int launch_cone_check(
    const float* d_cones,
    const float* d_points,
    float* d_scores,
    int n_cones,
    int n_points,
    float curvature,
    void* stream
) {
    // Validate inputs
    if (d_cones == nullptr || d_points == nullptr || d_scores == nullptr) {
        return cudaErrorInvalidValue;
    }

    if (n_cones <= 0 || n_points <= 0) {
        return cudaErrorInvalidValue;
    }

    // Validate curvature is negative (required for hyperbolic space)
    if (curvature >= 0.0f) {
        return cudaErrorInvalidValue;
    }

    // Check for NaN curvature
    if (isnan(curvature)) {
        return cudaErrorInvalidValue;
    }

    // Calculate grid dimensions
    // Block: 32 threads (points) × 8 cones = 256 threads per block
    // Grid: ceil(n_points/32) × ceil(n_cones/8) blocks
    dim3 block(BLOCK_DIM_X, BLOCK_DIM_Y);  // 32 x 8 = 256 threads
    dim3 grid(
        (n_points + BLOCK_DIM_X - 1) / BLOCK_DIM_X,
        (n_cones + CONES_PER_BLOCK - 1) / CONES_PER_BLOCK
    );

    // Launch kernel
    cudaStream_t cuda_stream = static_cast<cudaStream_t>(stream);
    cone_check_kernel<<<grid, block, 0, cuda_stream>>>(
        d_cones, d_points, d_scores,
        n_cones, n_points, curvature
    );

    // Check for kernel launch errors
    return static_cast<int>(cudaGetLastError());
}

/**
 * Single cone membership score (for API completeness).
 * Prefer batch version for efficiency - single computation has high overhead.
 *
 * @param d_cone      Device pointer to cone data [65]
 * @param d_point     Device pointer to point [64]
 * @param d_score     Device pointer to output score [1]
 * @param curvature   Poincare ball curvature (must be negative)
 * @param stream      CUDA stream (nullptr for default stream)
 * @return            0 (cudaSuccess) on success, CUDA error code on failure
 */
extern "C" int cone_check_single(
    const float* d_cone,
    const float* d_point,
    float* d_score,
    float curvature,
    void* stream
) {
    // Delegate to batch function with n=1
    return launch_cone_check(
        d_cone, d_point, d_score,
        1, 1, curvature, stream
    );
}

/**
 * Get kernel configuration info (for debugging/profiling).
 *
 * @param block_dim_x   Output: Block dimension X (32)
 * @param block_dim_y   Output: Block dimension Y (8)
 * @param point_dim     Output: Point dimension (64)
 * @param cone_data_dim Output: Cone data dimension (65 = 64 apex + 1 aperture)
 * @param shared_mem    Output: Shared memory per block in bytes
 */
extern "C" void get_cone_kernel_config(
    int* block_dim_x,
    int* block_dim_y,
    int* point_dim,
    int* cone_data_dim,
    int* shared_mem
) {
    if (block_dim_x) *block_dim_x = BLOCK_DIM_X;
    if (block_dim_y) *block_dim_y = BLOCK_DIM_Y;
    if (point_dim) *point_dim = POINT_DIM;
    if (cone_data_dim) *cone_data_dim = CONE_DATA_DIM;
    if (shared_mem) *shared_mem = sizeof(float) * CONES_PER_BLOCK * CONE_DATA_DIM;
}
