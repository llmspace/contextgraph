//! Core error types for the embedding pipeline.

use crate::types::{InputType, ModelId};
use thiserror::Error;

/// Comprehensive error type for all embedding pipeline failures.
///
/// # Error Categories
///
/// | Category | Variants | Recovery Strategy |
/// |----------|----------|-------------------|
/// | Model | ModelNotFound, ModelLoadError, NotInitialized | Retry with different config |
/// | Validation | InvalidDimension, InvalidValue, EmptyInput, InputTooLong | Fix input data |
/// | Processing | BatchError, FusionError, TokenizationError | Retry or fallback model |
/// | Infrastructure | GpuError, CacheError, IoError, Timeout | Retry or degrade |
/// | Configuration | ConfigError, UnsupportedModality | Fix configuration |
/// | Serialization | SerializationError | Fix data format |
///
/// # Design Principles
///
/// - **NO FALLBACKS**: Errors must propagate, not be silently handled
/// - **FAIL FAST**: Invalid state triggers immediate error
/// - **CONTEXTUAL**: Every variant includes debugging information
/// - **TRACEABLE**: Error chain preserved via `source`
#[derive(Debug, Error)]
pub enum EmbeddingError {
    // === Model Errors ===
    /// Model with given ID not registered in ModelRegistry.
    #[error("Model not found: {model_id:?}")]
    ModelNotFound { model_id: ModelId },

    /// Model weight loading failed (HuggingFace download, ONNX parse, etc).
    #[error("Model load failed for {model_id:?}: {source}")]
    ModelLoadError {
        model_id: ModelId,
        #[source]
        source: Box<dyn std::error::Error + Send + Sync>,
    },

    /// Model exists but embed() called before initialize().
    #[error("Model not initialized: {model_id:?}")]
    NotInitialized { model_id: ModelId },

    /// Model is already loaded in the registry.
    #[error("Model already loaded: {model_id:?}")]
    ModelAlreadyLoaded { model_id: ModelId },

    /// Model is not loaded in the registry.
    #[error("Model not loaded: {model_id:?}")]
    ModelNotLoaded { model_id: ModelId },

    /// Memory budget exceeded for loading models.
    #[error("Memory budget exceeded: requested {requested_bytes} bytes, available {available_bytes} bytes (budget: {budget_bytes} bytes)")]
    MemoryBudgetExceeded {
        requested_bytes: usize,
        available_bytes: usize,
        budget_bytes: usize,
    },

    /// Internal error (should not occur in normal operation).
    #[error("Internal error: {message}")]
    InternalError { message: String },

    // === Validation Errors ===
    /// Embedding vector dimension mismatch.
    #[error("Invalid dimension: expected {expected}, got {actual}")]
    InvalidDimension { expected: usize, actual: usize },

    /// Embedding contains NaN or Infinity at specific index.
    #[error("Invalid embedding value at index {index}: {value}")]
    InvalidValue { index: usize, value: f32 },

    /// Empty input provided (text, code, bytes).
    #[error("Empty input not allowed")]
    EmptyInput,

    /// Input exceeds model's max token limit.
    #[error("Input too long: {actual} tokens exceeds max {max}")]
    InputTooLong { actual: usize, max: usize },

    /// Invalid image data (decoding failed, corrupt, unsupported format).
    #[error("Invalid image: {reason}")]
    InvalidImage { reason: String },

    // === Processing Errors ===
    /// Batch processing failed (queue overflow, timeout, partial failure).
    #[error("Batch processing error: {message}")]
    BatchError { message: String },

    /// FuseMoE fusion failed (expert routing, gating, aggregation).
    #[error("Fusion error: {message}")]
    FusionError { message: String },

    /// Tokenization failed (unknown tokens, encoding error).
    #[error("Tokenization error for {model_id:?}: {message}")]
    TokenizationError { model_id: ModelId, message: String },

    // === Infrastructure Errors ===
    /// GPU/CUDA operation failed.
    #[error("GPU error: {message}")]
    GpuError { message: String },

    /// Embedding cache operation failed (LRU eviction, disk I/O).
    #[error("Cache error: {message}")]
    CacheError { message: String },

    /// File I/O error (model weights, config files).
    #[error("IO error: {0}")]
    IoError(#[from] std::io::Error),

    /// Operation exceeded timeout threshold.
    #[error("Operation timeout after {timeout_ms}ms")]
    Timeout { timeout_ms: u64 },

    // === Configuration Errors ===
    /// Model does not support the given input type.
    #[error("Unsupported input type {input_type:?} for model {model_id:?}")]
    UnsupportedModality {
        model_id: ModelId,
        input_type: InputType,
    },

    /// Configuration file invalid or missing required fields.
    #[error("Configuration error: {message}")]
    ConfigError { message: String },

    // === Serialization Errors ===
    /// Serialization/deserialization failed (JSON, binary, protobuf).
    #[error("Serialization error: {message}")]
    SerializationError { message: String },

    // === Expert Routing Errors ===
    /// Invalid expert index in FuseMoE routing.
    #[error("Invalid expert index: {index} (max: {max})")]
    InvalidExpertIndex { index: usize, max: usize },

    /// Dimension mismatch between expected and actual values.
    #[error("Dimension mismatch: expected {expected}, got {got}")]
    DimensionMismatch { expected: usize, got: usize },
}

/// Result type alias for embedding operations.
pub type EmbeddingResult<T> = Result<T, EmbeddingError>;
