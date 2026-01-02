//! Root configuration for the embedding pipeline.
//!
//! This module defines `EmbeddingConfig`, the top-level configuration struct
//! that aggregates all embedding subsystem configurations.
//!
//! # Loading Configuration
//!
//! ```rust,ignore
//! use context_graph_embeddings::EmbeddingConfig;
//!
//! // Load from file
//! let config = EmbeddingConfig::from_file("embeddings.toml")?;
//!
//! // Or use defaults for development
//! let config = EmbeddingConfig::default();
//!
//! // With environment overrides
//! let config = EmbeddingConfig::default().with_env_overrides();
//! ```
//!
//! # TOML Structure
//!
//! ```toml
//! [models]
//! models_dir = "./models"
//! lazy_loading = true
//! preload_models = ["semantic", "code"]
//!
//! [batch]
//! max_batch_size = 32
//! max_wait_ms = 50
//!
//! [fusion]
//! num_experts = 8
//! top_k = 4
//! output_dim = 1536
//!
//! [cache]
//! enabled = true
//! max_entries = 100000
//!
//! [gpu]
//! enabled = true
//! device_ids = [0]
//! ```
//!
//! # Design Principles
//!
//! - **NO FALLBACKS**: Invalid config returns error, never silently defaults
//! - **FAIL FAST**: File not found or parse error returns immediately
//! - **VALIDATION**: All nested configs are validated together

mod batch;
mod cache;
mod fusion;
mod gpu;
mod models;

#[cfg(test)]
mod tests;

// Re-export all public types for backwards compatibility
pub use batch::{BatchConfig, PaddingStrategy};
pub use cache::{CacheConfig, EvictionPolicy};
pub use fusion::FusionConfig;
pub use gpu::GpuConfig;
pub use models::ModelPathConfig;

use std::env;
use std::path::Path;

use serde::{Deserialize, Serialize};

use crate::error::{EmbeddingError, EmbeddingResult};

// ============================================================================
// ROOT EMBEDDING CONFIG
// ============================================================================

/// Root configuration for the embedding pipeline.
///
/// Aggregates all subsystem configurations.
/// Load from TOML file or use `Default::default()` for development.
///
/// # Example
///
/// ```rust,ignore
/// use context_graph_embeddings::EmbeddingConfig;
///
/// // Load from file
/// let config = EmbeddingConfig::from_file("config/embeddings.toml")?;
///
/// // Validate
/// config.validate()?;
///
/// // With environment overrides
/// let config = EmbeddingConfig::default().with_env_overrides();
/// ```
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct EmbeddingConfig {
    /// Model path configuration (paths, lazy loading, etc.)
    #[serde(default)]
    pub models: ModelPathConfig,

    /// Batch processing configuration
    #[serde(default)]
    pub batch: BatchConfig,

    /// FuseMoE fusion layer configuration
    #[serde(default)]
    pub fusion: FusionConfig,

    /// Embedding cache configuration
    #[serde(default)]
    pub cache: CacheConfig,

    /// GPU configuration
    #[serde(default)]
    pub gpu: GpuConfig,
}

impl EmbeddingConfig {
    /// Load configuration from a TOML file.
    ///
    /// # Arguments
    /// * `path` - Path to the TOML configuration file
    ///
    /// # Errors
    /// - `EmbeddingError::IoError` if file cannot be read
    /// - `EmbeddingError::ConfigError` if TOML parsing fails
    ///
    /// # Example
    ///
    /// ```rust,ignore
    /// let config = EmbeddingConfig::from_file("embeddings.toml")?;
    /// ```
    pub fn from_file(path: impl AsRef<Path>) -> EmbeddingResult<Self> {
        let path = path.as_ref();

        let contents = std::fs::read_to_string(path).map_err(|e| EmbeddingError::ConfigError {
            message: format!("Failed to read config file '{}': {}", path.display(), e),
        })?;

        let config: Self = toml::from_str(&contents).map_err(|e| EmbeddingError::ConfigError {
            message: format!("Failed to parse TOML in '{}': {}", path.display(), e),
        })?;

        Ok(config)
    }

    /// Validate all configuration values.
    ///
    /// Validates all nested configurations and returns the first error found.
    ///
    /// # Errors
    /// - `EmbeddingError::ConfigError` with descriptive message if any config is invalid
    ///
    /// # Example
    ///
    /// ```rust,ignore
    /// let config = EmbeddingConfig::default();
    /// config.validate()?; // Should pass for defaults
    /// ```
    pub fn validate(&self) -> EmbeddingResult<()> {
        // Validate each subsystem config, returning first error
        self.models
            .validate()
            .map_err(|e| EmbeddingError::ConfigError {
                message: format!("[models] {}", e),
            })?;

        self.batch
            .validate()
            .map_err(|e| EmbeddingError::ConfigError {
                message: format!("[batch] {}", e),
            })?;

        self.fusion
            .validate()
            .map_err(|e| EmbeddingError::ConfigError {
                message: format!("[fusion] {}", e),
            })?;

        self.cache
            .validate()
            .map_err(|e| EmbeddingError::ConfigError {
                message: format!("[cache] {}", e),
            })?;

        self.gpu
            .validate()
            .map_err(|e| EmbeddingError::ConfigError {
                message: format!("[gpu] {}", e),
            })?;

        Ok(())
    }

    /// Create configuration with environment variable overrides.
    ///
    /// Environment variables override TOML values. Prefix: `EMBEDDING_`
    ///
    /// # Supported Variables
    ///
    /// | Variable | Config Path | Type |
    /// |----------|-------------|------|
    /// | `EMBEDDING_MODELS_DIR` | `models.models_dir` | String |
    /// | `EMBEDDING_LAZY_LOADING` | `models.lazy_loading` | bool |
    /// | `EMBEDDING_GPU_ENABLED` | `gpu.enabled` | bool |
    /// | `EMBEDDING_CACHE_ENABLED` | `cache.enabled` | bool |
    /// | `EMBEDDING_CACHE_MAX_ENTRIES` | `cache.max_entries` | usize |
    /// | `EMBEDDING_BATCH_MAX_SIZE` | `batch.max_batch_size` | usize |
    /// | `EMBEDDING_BATCH_MAX_WAIT_MS` | `batch.max_wait_ms` | u64 |
    ///
    /// # Example
    ///
    /// ```rust,ignore
    /// std::env::set_var("EMBEDDING_GPU_ENABLED", "false");
    /// let config = EmbeddingConfig::default().with_env_overrides();
    /// assert!(!config.gpu.enabled);
    /// ```
    #[must_use]
    pub fn with_env_overrides(mut self) -> Self {
        // Models config
        if let Ok(val) = env::var("EMBEDDING_MODELS_DIR") {
            self.models.models_dir = val;
        }
        if let Ok(val) = env::var("EMBEDDING_LAZY_LOADING") {
            if let Ok(b) = val.parse::<bool>() {
                self.models.lazy_loading = b;
            }
        }

        // GPU config
        if let Ok(val) = env::var("EMBEDDING_GPU_ENABLED") {
            if let Ok(b) = val.parse::<bool>() {
                self.gpu.enabled = b;
            }
        }

        // Cache config
        if let Ok(val) = env::var("EMBEDDING_CACHE_ENABLED") {
            if let Ok(b) = val.parse::<bool>() {
                self.cache.enabled = b;
            }
        }
        if let Ok(val) = env::var("EMBEDDING_CACHE_MAX_ENTRIES") {
            if let Ok(n) = val.parse::<usize>() {
                self.cache.max_entries = n;
            }
        }

        // Batch config
        if let Ok(val) = env::var("EMBEDDING_BATCH_MAX_SIZE") {
            if let Ok(n) = val.parse::<usize>() {
                self.batch.max_batch_size = n;
            }
        }
        if let Ok(val) = env::var("EMBEDDING_BATCH_MAX_WAIT_MS") {
            if let Ok(n) = val.parse::<u64>() {
                self.batch.max_wait_ms = n;
            }
        }

        self
    }

    /// Create configuration from TOML string (for testing).
    ///
    /// # Errors
    /// - `EmbeddingError::ConfigError` if TOML parsing fails
    pub fn from_toml_str(toml: &str) -> EmbeddingResult<Self> {
        toml::from_str(toml).map_err(|e| EmbeddingError::ConfigError {
            message: format!("Failed to parse TOML: {}", e),
        })
    }

    /// Serialize configuration to TOML string.
    ///
    /// # Errors
    /// - `EmbeddingError::ConfigError` if serialization fails
    pub fn to_toml_string(&self) -> EmbeddingResult<String> {
        toml::to_string_pretty(self).map_err(|e| EmbeddingError::ConfigError {
            message: format!("Failed to serialize to TOML: {}", e),
        })
    }
}
