//! GpuTensor wrapper for type-safe GPU tensor operations.
//!
//! # Design
//!
//! `GpuTensor` wraps `candle_core::Tensor` with additional tracking:
//! - Automatic device placement
//! - Memory usage tracking
//! - Easy conversion to/from CPU vectors
//!
//! # Usage
//!
//! ```rust,ignore
//! use context_graph_embeddings::gpu::GpuTensor;
//!
//! // Create from CPU vector
//! let vec = vec![1.0f32, 2.0, 3.0, 4.0];
//! let tensor = GpuTensor::from_vec(&vec)?;
//!
//! // Perform GPU operations
//! let normalized = tensor.normalize()?;
//!
//! // Convert back to CPU
//! let result: Vec<f32> = normalized.to_vec()?;
//! ```

mod core;
mod ops;

pub use self::core::GpuTensor;
