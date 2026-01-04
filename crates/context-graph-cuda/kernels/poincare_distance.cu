// CUDA kernel for batch Poincare ball distance computation
// Target: RTX 5090 (Compute Capability 12.0, CUDA 13.1)
// Performance: <1ms for 1K x 1K distance matrix
//
// Constitution Reference:
// - TECH-GRAPH-004 Section 10.1: Poincare CUDA Kernel
// - perf.latency.poincare_distance_gpu: <1ms for 1K x 1K
// - stack.gpu: RTX 5090, 32GB GDDR7, 1792 GB/s
//
// Mathematics:
// Poincare ball distance formula (direct computation, GPU-friendly):
// d(x,y) = (2/sqrt(c)) * arctanh(sqrt(c * ||x-y||² / ((1-c*||x||²)(1-c*||y||²))))
// where c = |curvature| (always positive internally, since hyperbolic space has negative curvature)
//
// This is mathematically equivalent to:
// d(x,y) = (2/sqrt(c)) * arctanh(sqrt(c) * ||(-x) ⊕ y||)
// but avoids Mobius addition intermediate allocation.

#include <cuda_runtime.h>
#include <math.h>

// Kernel configuration - tuned for RTX 5090 (170 SMs, 21760 CUDA cores)
// Block: 32 x 8 = 256 threads (optimal for Blackwell)
// Shared memory: 8 queries × 64 floats × 4 bytes + 8 norms × 4 bytes = ~2KB per block
constexpr int BLOCK_DIM_X = 32;       // Warp size for coalesced access
constexpr int BLOCK_DIM_Y = 8;        // 8 queries per block
constexpr int POINT_DIM = 64;         // Poincare ball dimension (fixed, SIMD-aligned)
constexpr int QUERIES_PER_BLOCK = 8;  // Matches BLOCK_DIM_Y

// Numerical stability constants
constexpr float EPS = 1e-7f;          // Prevent division by zero
constexpr float ARCTANH_CLAMP = 1.0f - 1e-7f;  // arctanh domain: (-1, 1)

/**
 * Compute batch Poincare ball distances on GPU.
 *
 * Grid: (ceil(n_database/32), ceil(n_queries/8))
 * Block: (32, 8) = 256 threads
 *
 * Each thread computes one (query, database) distance pair.
 * Shared memory caches query vectors for reuse across database points.
 *
 * Memory access pattern:
 * - Queries: Cached in shared memory (reused by all threads in a row)
 * - Database: Coalesced global reads (adjacent threads read adjacent elements)
 * - Output: Coalesced writes (row-major order)
 *
 * @param queries     [n_queries][64] row-major, device memory
 * @param database    [n_database][64] row-major, device memory
 * @param distances   [n_queries][n_database] row-major, device memory (OUTPUT)
 * @param n_queries   Number of query points
 * @param n_database  Number of database points
 * @param curvature   Poincare ball curvature (must be negative, typically -1.0)
 */
__global__ void poincare_distance_kernel(
    const float* __restrict__ queries,
    const float* __restrict__ database,
    float* __restrict__ distances,
    int n_queries,
    int n_database,
    float curvature
) {
    // Shared memory for query vectors and their squared norms
    // Query caching provides ~8x reuse per thread block
    __shared__ float shared_queries[QUERIES_PER_BLOCK][POINT_DIM];
    __shared__ float shared_query_norms_sq[QUERIES_PER_BLOCK];

    const int tx = threadIdx.x;  // 0-31: database point index within tile
    const int ty = threadIdx.y;  // 0-7: query index within block
    const int bx = blockIdx.x;   // Database tile index
    const int by = blockIdx.y;   // Query tile index

    const int query_idx = by * QUERIES_PER_BLOCK + ty;
    const int db_idx = bx * BLOCK_DIM_X + tx;

    // Curvature magnitude and precomputed constants
    // For hyperbolic space: c = |curvature| (always positive)
    const float c = fabsf(curvature);
    const float sqrt_c = sqrtf(c);
    const float two_over_sqrt_c = 2.0f / sqrt_c;

    // =========================================================================
    // Phase 1: Load query vectors into shared memory
    // =========================================================================
    // Each thread in a row collaboratively loads the query vector
    // Thread tx loads elements d, d+32, d+64, ... (strided for coalescing)
    if (query_idx < n_queries) {
        for (int d = tx; d < POINT_DIM; d += BLOCK_DIM_X) {
            shared_queries[ty][d] = queries[query_idx * POINT_DIM + d];
        }
    }
    __syncthreads();

    // =========================================================================
    // Phase 2: Compute query norms (parallel reduction per query)
    // =========================================================================
    // Only thread 0 in each row computes the query norm
    // This is sufficient since we cache the result in shared memory
    if (query_idx < n_queries && tx == 0) {
        float norm_sq = 0.0f;
        #pragma unroll 8
        for (int d = 0; d < POINT_DIM; d++) {
            float val = shared_queries[ty][d];
            norm_sq += val * val;
        }
        shared_query_norms_sq[ty] = norm_sq;
    }
    __syncthreads();

    // =========================================================================
    // Phase 3: Compute distance for this (query, database) pair
    // =========================================================================
    if (query_idx < n_queries && db_idx < n_database) {
        // Load database point into registers for fast access
        // Each thread loads one complete database point (256 bytes)
        float db_norm_sq = 0.0f;
        float diff_norm_sq = 0.0f;

        // Fused loop: Load database point, compute its norm, and compute diff norm
        #pragma unroll 8
        for (int d = 0; d < POINT_DIM; d++) {
            float db_val = database[db_idx * POINT_DIM + d];
            float q_val = shared_queries[ty][d];

            db_norm_sq += db_val * db_val;

            float diff = q_val - db_val;
            diff_norm_sq += diff * diff;
        }

        // Get cached query norm
        float query_norm_sq = shared_query_norms_sq[ty];

        // Denominators: (1 - c*||x||²) and (1 - c*||y||²)
        // Clamp to epsilon to avoid division by zero when points are at boundary
        float denom_x = fmaxf(1.0f - c * query_norm_sq, EPS);
        float denom_y = fmaxf(1.0f - c * db_norm_sq, EPS);

        // Compute arctanh argument: sqrt(c * ||x-y||² / (denom_x * denom_y))
        float arg_squared = c * diff_norm_sq / (denom_x * denom_y);
        float arg = sqrtf(fmaxf(arg_squared, 0.0f));  // Ensure non-negative for sqrt

        // Clamp to valid arctanh domain: (-1, 1)
        // Near 1.0, arctanh approaches infinity, so we clamp slightly below
        arg = fminf(arg, ARCTANH_CLAMP);

        // Poincare distance: (2/sqrt(c)) * arctanh(arg)
        float dist = two_over_sqrt_c * atanhf(arg);

        // Write result to global memory (coalesced write)
        distances[query_idx * n_database + db_idx] = dist;
    }
}

/**
 * Host function to launch Poincare distance kernel.
 *
 * Automatically calculates optimal grid dimensions based on input sizes.
 * Validates curvature is negative (required for hyperbolic space).
 *
 * @param d_queries    Device pointer to query vectors [n_queries][64]
 * @param d_database   Device pointer to database vectors [n_database][64]
 * @param d_distances  Device pointer to output distance matrix [n_queries][n_database]
 * @param n_queries    Number of query points
 * @param n_database   Number of database points
 * @param curvature    Poincare ball curvature (must be negative)
 * @param stream       CUDA stream (nullptr for default stream)
 * @return cudaError_t Error code (cudaSuccess on success)
 */
extern "C" cudaError_t launch_poincare_distance(
    const float* d_queries,
    const float* d_database,
    float* d_distances,
    int n_queries,
    int n_database,
    float curvature,
    cudaStream_t stream
) {
    // Validate inputs
    if (d_queries == nullptr || d_database == nullptr || d_distances == nullptr) {
        return cudaErrorInvalidValue;
    }

    if (n_queries <= 0 || n_database <= 0) {
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
    // Block: 32 threads × 8 queries = 256 threads per block
    // Grid: ceil(n_database/32) × ceil(n_queries/8) blocks
    dim3 block(BLOCK_DIM_X, BLOCK_DIM_Y);  // 32 x 8 = 256 threads
    dim3 grid(
        (n_database + BLOCK_DIM_X - 1) / BLOCK_DIM_X,
        (n_queries + QUERIES_PER_BLOCK - 1) / QUERIES_PER_BLOCK
    );

    // Launch kernel
    poincare_distance_kernel<<<grid, block, 0, stream>>>(
        d_queries, d_database, d_distances,
        n_queries, n_database, curvature
    );

    // Check for kernel launch errors
    return cudaGetLastError();
}

/**
 * Single-pair Poincare distance (for API completeness).
 * Prefer batch version for efficiency - single-pair has high overhead.
 *
 * @param d_point_a   Device pointer to first point [64]
 * @param d_point_b   Device pointer to second point [64]
 * @param d_distance  Device pointer to output distance [1]
 * @param curvature   Poincare ball curvature (must be negative)
 * @param stream      CUDA stream (nullptr for default stream)
 * @return cudaError_t Error code
 */
extern "C" cudaError_t poincare_distance_single(
    const float* d_point_a,
    const float* d_point_b,
    float* d_distance,
    float curvature,
    cudaStream_t stream
) {
    // Delegate to batch function with n=1
    return launch_poincare_distance(
        d_point_a, d_point_b, d_distance,
        1, 1, curvature, stream
    );
}

/**
 * Get kernel configuration info (for debugging/profiling).
 *
 * @param block_dim_x   Output: Block dimension X (32)
 * @param block_dim_y   Output: Block dimension Y (8)
 * @param point_dim     Output: Point dimension (64)
 * @param shared_mem    Output: Shared memory per block in bytes
 */
extern "C" void get_poincare_kernel_config(
    int* block_dim_x,
    int* block_dim_y,
    int* point_dim,
    int* shared_mem
) {
    if (block_dim_x) *block_dim_x = BLOCK_DIM_X;
    if (block_dim_y) *block_dim_y = BLOCK_DIM_Y;
    if (point_dim) *point_dim = POINT_DIM;
    if (shared_mem) *shared_mem = sizeof(float) * (QUERIES_PER_BLOCK * POINT_DIM + QUERIES_PER_BLOCK);
}
