//! CUDA acceleration for Context Graph.
//!
//! This crate provides GPU-accelerated operations for:
//! - Vector similarity search (cosine, dot product)
//! - Neural attention mechanisms
//! - Modern Hopfield network computations
//!
//! For Phase 0 (Ghost System), stub implementations run on CPU.
//! Future phases will use cudarc bindings for RTX 5090 (Blackwell) optimization.
//!
//! # Target Hardware
//!
//! - RTX 5090 (32GB GDDR7, 1.8 TB/s bandwidth)
//! - CUDA 13.1 with Compute Capability 12.0
//! - Blackwell architecture optimizations
//!
//! # Example
//!
//! ```
//! use context_graph_cuda::{StubVectorOps, VectorOps};
//!
//! // Create stub vector ops for CPU fallback
//! let ops = StubVectorOps::new();
//! // Stub uses CPU, so GPU is not available
//! assert!(!ops.is_gpu_available());
//! assert_eq!(ops.device_name(), "CPU (Stub)");
//! ```

pub mod cone;
pub mod error;
pub mod ops;
pub mod poincare;
pub mod stub;

pub use error::{CudaError, CudaResult};
pub use ops::VectorOps;
pub use poincare::{PoincareCudaConfig, poincare_distance_cpu, poincare_distance_batch_cpu};
#[cfg(feature = "cuda")]
pub use poincare::{poincare_distance_batch_gpu, poincare_distance_single_gpu};
pub use cone::{
    ConeCudaConfig, ConeData, ConeKernelInfo,
    cone_check_batch_cpu, cone_membership_score_cpu,
    is_cone_gpu_available, get_cone_kernel_info,
    CONE_DATA_DIM, POINT_DIM,
};
#[cfg(feature = "cuda")]
pub use cone::{cone_check_batch_gpu, cone_check_single_gpu};
pub use stub::StubVectorOps;
