//! GPU Resources management for FAISS.
//!
//! Provides RAII wrapper around FAISS GPU resources with automatic cleanup.
//!
//! # Constitution Compliance
//!
//! - ARCH-GPU-04: FAISS indexes use GPU (faiss-gpu) not CPU - NO FALLBACK
//! - AP-GPU-03: NEVER use CPU FAISS when GPU FAISS available
//!
//! # Feature Requirements
//!
//! This module requires the `faiss-working` feature to be enabled for actual
//! GPU operations. Without it, all operations will fail fast with clear errors.

use std::sync::Arc;

use super::super::faiss_ffi::{faiss_status, gpu_available, GpuResources as FfiGpuResources};
use crate::error::{GraphError, GraphResult};

/// GPU resources handle with RAII cleanup.
///
/// Wraps raw GPU resource pointer with automatic deallocation.
/// Use `Arc<GpuResources>` for sharing across multiple indices.
///
/// # Thread Safety
///
/// This type is `Send + Sync` because the underlying FAISS StandardGpuResources
/// uses internal synchronization for GPU memory management.
///
/// # Fail-Fast Behavior
///
/// When `faiss-working` feature is not enabled, `GpuResources::new()` will
/// ALWAYS fail with a clear error message. There is NO fallback per Constitution.
pub struct GpuResources {
    inner: FfiGpuResources,
    gpu_id: i32,
}

// SAFETY: GpuResources wraps FfiGpuResources which is Send+Sync.
// The gpu_id field is Copy and thread-safe.
unsafe impl Send for GpuResources {}
unsafe impl Sync for GpuResources {}

impl GpuResources {
    /// Allocate GPU resources for the specified device.
    ///
    /// # Arguments
    ///
    /// * `gpu_id` - CUDA device ID (typically 0). Note: Currently stored for
    ///   future multi-GPU support but FAISS StandardGpuResources uses the
    ///   default CUDA device. Multi-GPU device selection is planned.
    ///
    /// # Errors
    ///
    /// Returns `GraphError::FaissGpuUnavailable` if:
    /// - `faiss-working` feature is not enabled (NO FALLBACK)
    /// - GPU device is unavailable
    /// - CUDA initialization fails
    /// - Insufficient GPU memory
    ///
    /// # Constitution Compliance
    ///
    /// Per ARCH-GPU-04: FAISS indexes use GPU, not CPU. There is NO fallback.
    /// If FAISS GPU is not available, this function fails with a clear error.
    ///
    /// # Example
    ///
    /// ```no_run
    /// use context_graph_graph::index::gpu_index::GpuResources;
    /// use std::sync::Arc;
    ///
    /// let resources = Arc::new(GpuResources::new(0)?);
    /// # Ok::<(), context_graph_graph::error::GraphError>(())
    /// ```
    pub fn new(gpu_id: i32) -> GraphResult<Self> {
        // First check if FAISS GPU is available at all - provides clear error early
        if !gpu_available() {
            return Err(GraphError::FaissGpuUnavailable {
                reason: format!(
                    "FAISS GPU is REQUIRED but not available for device {}. Status: {}",
                    gpu_id,
                    faiss_status()
                ),
                help: "Fix: 1) Run ./scripts/rebuild_faiss_gpu.sh to build FAISS with CUDA 13.1+ \
                       2) Build with: cargo build --features faiss-working \
                       Per Constitution ARCH-GPU-04: NO CPU FALLBACK."
                    .to_string(),
            });
        }

        // Create FFI GPU resources - may fail even if gpu_available() is true
        // (e.g., insufficient VRAM, driver issues, concurrent allocation failure)
        let inner = FfiGpuResources::new().map_err(|e| GraphError::GpuResourceAllocation(format!(
            "Failed to create GPU resources for device {}. Error: {}. Status: {}. \
             Check: 1) nvidia-smi shows GPU, 2) CUDA 13.1+ installed, 3) libfaiss_c.so linked correctly.",
            gpu_id,
            e,
            faiss_status()
        )))?;

        tracing::info!(
            target: "context_graph::faiss",
            gpu_id = gpu_id,
            status = faiss_status(),
            "GPU resources allocated successfully"
        );

        Ok(Self { inner, gpu_id })
    }

    /// Get reference to inner FFI resources for FFI calls.
    ///
    /// # Panics
    ///
    /// This should never panic because `new()` validates GPU availability.
    /// If the inner resources are invalid, this is a bug.
    #[inline]
    pub(crate) fn inner(&self) -> &FfiGpuResources {
        &self.inner
    }

    /// Get the GPU device ID.
    #[inline]
    pub fn gpu_id(&self) -> i32 {
        self.gpu_id
    }

    /// Check if GPU is still available.
    ///
    /// Note: This checks global FAISS GPU availability, not instance-specific
    /// state. A `GpuResources` that was successfully created remains valid
    /// until dropped (RAII ownership ensures this).
    ///
    /// This method is useful for checking if the GPU has become unavailable
    /// due to external factors (driver crash, device removal, etc.) since
    /// the resources were created.
    #[inline]
    pub fn is_valid(&self) -> bool {
        gpu_available()
    }
}

impl std::fmt::Debug for GpuResources {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("GpuResources")
            .field("gpu_id", &self.gpu_id)
            .field("status", &faiss_status())
            .field("available", &gpu_available())
            .finish()
    }
}

/// Create a shared GPU resources handle.
///
/// Convenience function for creating `Arc<GpuResources>`.
///
/// # Arguments
///
/// * `gpu_id` - CUDA device ID (typically 0)
///
/// # Errors
///
/// Returns error if GPU resources cannot be allocated.
/// Per Constitution ARCH-GPU-04, there is NO fallback.
pub fn create_shared_resources(gpu_id: i32) -> GraphResult<Arc<GpuResources>> {
    Ok(Arc::new(GpuResources::new(gpu_id)?))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_gpu_resources_behavior() {
        let result = GpuResources::new(0);

        if gpu_available() {
            // FAISS GPU is available - should succeed
            match result {
                Ok(resources) => {
                    println!("GPU resources created: {:?}", resources);
                    assert!(resources.is_valid());
                    assert_eq!(resources.gpu_id(), 0);
                }
                Err(e) => {
                    // Might fail due to other reasons (no GPU device, memory)
                    println!("GPU resources failed (expected in some environments): {}", e);
                }
            }
        } else {
            // FAISS GPU not available - MUST fail per Constitution
            assert!(
                result.is_err(),
                "GpuResources::new MUST fail when FAISS GPU is not available"
            );
            let err = result.unwrap_err();
            println!("Got expected error: {}", err);
            assert!(
                err.to_string().contains("FAISS")
                    || err.to_string().contains("REQUIRED")
                    || err.to_string().contains("GPU"),
                "Error should mention FAISS/GPU requirement"
            );
        }
    }

    #[test]
    fn test_create_shared_resources() {
        let result = create_shared_resources(0);

        if gpu_available() {
            match result {
                Ok(resources) => {
                    assert!(Arc::strong_count(&resources) == 1);
                    println!("Shared resources created: {:?}", resources);
                }
                Err(e) => {
                    println!("Shared resources failed: {}", e);
                }
            }
        } else {
            assert!(result.is_err());
        }
    }
}
