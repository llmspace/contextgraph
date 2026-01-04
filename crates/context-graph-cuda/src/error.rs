//! Error types for CUDA operations.

use thiserror::Error;

/// CUDA-specific errors.
#[derive(Debug, Error)]
pub enum CudaError {
    /// CUDA device initialization failed.
    #[error("Failed to initialize CUDA device: {0}")]
    DeviceInitError(String),

    /// Memory allocation failed.
    #[error("CUDA memory allocation failed: {0}")]
    MemoryError(String),

    /// Kernel execution failed.
    #[error("CUDA kernel execution failed: {0}")]
    KernelError(String),

    /// Dimension mismatch.
    #[error("Dimension mismatch: expected {expected}, got {actual}")]
    DimensionMismatch { expected: usize, actual: usize },

    /// Device not available.
    #[error("No CUDA device available")]
    NoDevice,

    /// Feature not implemented.
    #[error("Feature not implemented: {0}")]
    NotImplemented(String),

    /// Invalid configuration parameter.
    #[error("Invalid configuration: {0}")]
    InvalidConfig(String),
}

/// Result type for CUDA operations.
pub type CudaResult<T> = Result<T, CudaError>;
