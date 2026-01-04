//! Pipeline Integration Module for Warm Model Loading
//!
//! Connects [`WarmLoader`] with the embedding pipeline to provide a unified
//! warmed embedding system with all models pre-loaded in VRAM at startup.
//!
//! # Critical Design Decisions
//!
//! ## NO WORKAROUNDS OR FALLBACKS
//!
//! This integration implements a **fail-fast** strategy. If any component
//! fails during initialization, the pipeline terminates immediately with
//! an appropriate exit code. There are no:
//!
//! - Partial initialization modes
//! - Degraded operation states
//! - Mock or fallback models
//!
//! ## Exit Behavior
//!
//! On fatal errors, [`WarmEmbeddingPipeline::create_and_warm()`] calls
//! `std::process::exit()` with the error's exit code (101-110).
//!
//! # Requirements Implemented
//!
//! - REQ-WARM-001: Load all 12 embedding models at startup
//! - REQ-WARM-002: Load FuseMoE layer at startup
//! - REQ-WARM-003: Validate models with test inference
//! - REQ-WARM-006: Health check status reporting
//! - REQ-WARM-007: Per-model state visibility
//!
//! # Example
//!
//! ```rust,ignore
//! use context_graph_embeddings::warm::{WarmConfig, WarmEmbeddingPipeline};
//!
//! // Production usage - fails fast on any error
//! let pipeline = WarmEmbeddingPipeline::create_and_warm(WarmConfig::default())?;
//!
//! // Check readiness
//! assert!(pipeline.is_ready());
//!
//! // Access health status
//! let health = pipeline.health();
//! println!("Status: {:?}, Models: {}/{}",
//!     health.status, health.models_warm, health.models_total);
//!
//! // Access registry for model handles
//! let registry = pipeline.registry();
//! ```

use std::sync::atomic::{AtomicBool, Ordering};
use std::time::{Duration, Instant};

use super::config::WarmConfig;
use super::diagnostics::{WarmDiagnosticReport, WarmDiagnostics};
use super::error::{WarmError, WarmResult};
use super::health::{WarmHealthCheck, WarmHealthChecker};
use super::loader::WarmLoader;
use super::registry::SharedWarmRegistry;

// ============================================================================
// WarmEmbeddingPipeline
// ============================================================================

/// Warmed embedding pipeline with all models pre-loaded in VRAM.
///
/// This is the main entry point for production use of the warm loading system.
/// It integrates [`WarmLoader`], [`WarmHealthChecker`], and [`WarmDiagnostics`]
/// into a unified pipeline that:
///
/// - Loads all 13 models (12 embeddings + FuseMoE) into VRAM at startup
/// - Validates each model with test inference
/// - Provides real-time health monitoring
/// - Generates diagnostic reports on demand
///
/// # Fail-Fast Behavior
///
/// The [`create_and_warm()`](Self::create_and_warm) method will terminate the
/// process immediately on any initialization error. This ensures predictable
/// behavior: the system is either fully operational or not running at all.
///
/// # Thread Safety
///
/// The pipeline is thread-safe for read operations. Health checks, diagnostics,
/// and registry access can be performed concurrently from multiple threads.
///
/// # Example
///
/// ```rust,ignore
/// // Create and warm the pipeline (production usage)
/// let pipeline = WarmEmbeddingPipeline::create_and_warm(WarmConfig::default())?;
///
/// // The pipeline is now ready for inference
/// assert!(pipeline.is_ready());
///
/// // Get uptime since initialization
/// if let Some(uptime) = pipeline.uptime() {
///     println!("Pipeline running for {:?}", uptime);
/// }
/// ```
pub struct WarmEmbeddingPipeline {
    /// Main orchestrator for warm model loading.
    loader: WarmLoader,
    /// Health check service for status monitoring.
    health_checker: WarmHealthChecker,
    /// Whether the pipeline has been successfully initialized.
    initialized: AtomicBool,
    /// Timestamp when initialization completed successfully.
    initialization_time: Option<Instant>,
}

impl WarmEmbeddingPipeline {
    /// Create and warm the embedding pipeline.
    ///
    /// This is the main entry point for production use. It creates the loader,
    /// loads all models, validates them, and sets up health monitoring.
    ///
    /// # Fail-Fast Behavior
    ///
    /// On ANY initialization error, this method:
    /// 1. Logs comprehensive diagnostic information
    /// 2. Dumps diagnostics to stderr
    /// 3. Calls `std::process::exit()` with the appropriate exit code
    ///
    /// It only returns `Ok(Self)` on complete success.
    ///
    /// # Arguments
    ///
    /// * `config` - Configuration for VRAM budgets, paths, and behavior
    ///
    /// # Returns
    ///
    /// `Ok(Self)` if all models are successfully loaded and validated.
    /// Never returns `Err` - fatal errors exit the process.
    ///
    /// # Example
    ///
    /// ```rust,ignore
    /// let config = WarmConfig::from_env();
    /// let pipeline = WarmEmbeddingPipeline::create_and_warm(config)?;
    /// assert!(pipeline.is_ready());
    /// ```
    pub fn create_and_warm(config: WarmConfig) -> WarmResult<Self> {
        tracing::info!("Starting WarmEmbeddingPipeline initialization");
        let start_time = Instant::now();

        // Step 1: Create loader
        tracing::info!("Step 1/4: Creating WarmLoader");
        let mut loader = match WarmLoader::new(config) {
            Ok(l) => l,
            Err(e) => {
                tracing::error!(
                    exit_code = e.exit_code(),
                    category = %e.category(),
                    "Failed to create WarmLoader"
                );
                Self::handle_fatal_error_static(&e);
            }
        };

        // Step 2: Load all models (FAIL FAST on any error)
        tracing::info!("Step 2/4: Loading all models into VRAM");
        if let Err(e) = loader.load_all_models() {
            // load_all_models already handles fatal errors internally,
            // but if it somehow returns an error, handle it here too
            tracing::error!(
                exit_code = e.exit_code(),
                category = %e.category(),
                "Failed to load models"
            );
            WarmDiagnostics::dump_to_stderr(&loader);
            Self::handle_fatal_error_static(&e);
        }

        // Step 3: Create health checker
        tracing::info!("Step 3/4: Creating health checker");
        let health_checker = WarmHealthChecker::from_loader(&loader);

        // Step 4: Verify health
        tracing::info!("Step 4/4: Verifying pipeline health");
        if !health_checker.is_healthy() {
            let health = health_checker.check();
            let error = WarmError::ModelValidationFailed {
                model_id: "pipeline".to_string(),
                reason: format!(
                    "Pipeline unhealthy after loading: {} warm, {} failed, {} loading",
                    health.models_warm, health.models_failed, health.models_loading
                ),
                expected_output: Some(format!("Healthy ({} models warm)", health.models_total)),
                actual_output: Some(format!("{:?}", health.status)),
            };

            tracing::error!(
                status = ?health.status,
                models_warm = health.models_warm,
                models_failed = health.models_failed,
                "Pipeline health check failed"
            );

            for msg in &health.error_messages {
                tracing::error!(error = %msg, "Model failure");
            }

            WarmDiagnostics::dump_to_stderr(&loader);
            Self::handle_fatal_error_static(&error);
        }

        let duration = start_time.elapsed();
        tracing::info!(
            duration_ms = duration.as_millis() as u64,
            "WarmEmbeddingPipeline initialization completed successfully"
        );

        Ok(Self {
            loader,
            health_checker,
            initialized: AtomicBool::new(true),
            initialization_time: Some(Instant::now()),
        })
    }

    /// Create a new pipeline without warming (for testing).
    ///
    /// Unlike [`create_and_warm()`](Self::create_and_warm), this method does
    /// NOT load models or exit on error. Use this when you need to test
    /// pipeline behavior without the full loading process.
    ///
    /// # Arguments
    ///
    /// * `config` - Configuration for the pipeline
    ///
    /// # Returns
    ///
    /// `Ok(Self)` with an uninitialized pipeline, or `Err` on configuration error.
    ///
    /// # Example
    ///
    /// ```rust,ignore
    /// let pipeline = WarmEmbeddingPipeline::new(WarmConfig::default())?;
    /// assert!(!pipeline.is_ready()); // Not ready until warmed
    ///
    /// // Manually warm when needed
    /// pipeline.warm()?;
    /// assert!(pipeline.is_ready());
    /// ```
    pub fn new(config: WarmConfig) -> WarmResult<Self> {
        tracing::debug!("Creating WarmEmbeddingPipeline without warming");

        let loader = WarmLoader::new(config)?;
        let health_checker = WarmHealthChecker::from_loader(&loader);

        Ok(Self {
            loader,
            health_checker,
            initialized: AtomicBool::new(false),
            initialization_time: None,
        })
    }

    /// Warm all models (call after [`new()`](Self::new) if used).
    ///
    /// Loads all 13 models into VRAM and validates them. Unlike
    /// [`create_and_warm()`](Self::create_and_warm), this method returns
    /// an error instead of exiting on failure.
    ///
    /// # Errors
    ///
    /// Returns `WarmError` if any model fails to load or validate.
    /// Note: The underlying loader may still call `exit()` on certain
    /// fatal errors (CUDA context loss, etc.).
    ///
    /// # Example
    ///
    /// ```rust,ignore
    /// let mut pipeline = WarmEmbeddingPipeline::new(config)?;
    /// pipeline.warm()?;
    /// assert!(pipeline.is_ready());
    /// ```
    pub fn warm(&mut self) -> WarmResult<()> {
        if self.initialized.load(Ordering::SeqCst) {
            tracing::warn!("Pipeline already initialized, skipping warm()");
            return Ok(());
        }

        tracing::info!("Warming pipeline models");

        // Load all models
        self.loader.load_all_models()?;

        // Update health checker with fresh data
        self.health_checker = WarmHealthChecker::from_loader(&self.loader);

        // Verify health
        if !self.health_checker.is_healthy() {
            let health = self.health_checker.check();
            return Err(WarmError::ModelValidationFailed {
                model_id: "pipeline".to_string(),
                reason: format!(
                    "Pipeline unhealthy: {} warm, {} failed",
                    health.models_warm, health.models_failed
                ),
                expected_output: Some("Healthy".to_string()),
                actual_output: Some(format!("{:?}", health.status)),
            });
        }

        self.initialized.store(true, Ordering::SeqCst);
        self.initialization_time = Some(Instant::now());

        tracing::info!("Pipeline warming completed successfully");
        Ok(())
    }

    /// Check if the pipeline is ready for inference.
    ///
    /// Returns `true` only if:
    /// - The pipeline has been initialized (via `create_and_warm()` or `warm()`)
    /// - All models are in the Warm state
    /// - The health checker reports Healthy status
    ///
    /// # Thread Safety
    ///
    /// Safe for concurrent access from multiple threads.
    ///
    /// # Example
    ///
    /// ```rust,ignore
    /// if pipeline.is_ready() {
    ///     // Safe to run inference
    ///     let embeddings = compute_embeddings(&pipeline, input);
    /// } else {
    ///     // Wait or fail gracefully
    ///     eprintln!("Pipeline not ready");
    /// }
    /// ```
    #[must_use]
    pub fn is_ready(&self) -> bool {
        self.initialized.load(Ordering::SeqCst) && self.health_checker.is_healthy()
    }

    /// Get the current health status.
    ///
    /// Returns a detailed [`WarmHealthCheck`] containing:
    /// - Overall health status (Healthy, Loading, Unhealthy, NotInitialized)
    /// - Model counts by state (warm, loading, failed, pending)
    /// - VRAM allocation metrics
    /// - Working memory metrics
    /// - Error messages from failed models
    ///
    /// # Thread Safety
    ///
    /// Safe for concurrent access. Acquires a read lock on the registry.
    ///
    /// # Example
    ///
    /// ```rust,ignore
    /// let health = pipeline.health();
    /// match health.status {
    ///     WarmHealthStatus::Healthy => println!("All {} models ready", health.models_total),
    ///     WarmHealthStatus::Unhealthy => {
    ///         for err in &health.error_messages {
    ///             eprintln!("Error: {}", err);
    ///         }
    ///     }
    ///     _ => println!("Status: {:?}", health.status),
    /// }
    /// ```
    #[must_use]
    pub fn health(&self) -> WarmHealthCheck {
        self.health_checker.check()
    }

    /// Get a diagnostic report for the pipeline.
    ///
    /// Generates a comprehensive [`WarmDiagnosticReport`] containing:
    /// - System information (hostname, OS)
    /// - GPU information (device, VRAM, compute capability)
    /// - Memory pool status (model pool, working pool)
    /// - Per-model state and VRAM allocations
    /// - Any errors encountered
    ///
    /// The report is JSON-serializable for automated monitoring.
    ///
    /// # Example
    ///
    /// ```rust,ignore
    /// let report = pipeline.diagnostics();
    /// println!("Warm models: {}/{}", report.warm_count(), report.models.len());
    ///
    /// // Serialize to JSON for monitoring
    /// let json = serde_json::to_string_pretty(&report)?;
    /// ```
    #[must_use]
    pub fn diagnostics(&self) -> WarmDiagnosticReport {
        WarmDiagnostics::generate_report(&self.loader)
    }

    /// Get a reference to the underlying loader.
    ///
    /// For advanced use cases requiring direct access to the loader.
    ///
    /// # Example
    ///
    /// ```rust,ignore
    /// let loader = pipeline.loader();
    /// let summary = loader.loading_summary();
    /// println!("VRAM allocated: {}", summary.vram_allocated_string());
    /// ```
    #[must_use]
    pub fn loader(&self) -> &WarmLoader {
        &self.loader
    }

    /// Get a reference to the shared registry.
    ///
    /// The registry provides access to model handles for inference.
    /// Use `read()` to acquire a read lock for concurrent access.
    ///
    /// # Example
    ///
    /// ```rust,ignore
    /// let registry = pipeline.registry();
    /// let guard = registry.read().unwrap();
    ///
    /// if let Some(handle) = guard.get_handle("E1_Semantic") {
    ///     println!("Model at VRAM address: {}", handle.vram_address_hex());
    /// }
    /// ```
    #[must_use]
    pub fn registry(&self) -> &SharedWarmRegistry {
        self.loader.registry()
    }

    /// Get the uptime since successful initialization.
    ///
    /// Returns `None` if the pipeline has not been initialized.
    ///
    /// # Example
    ///
    /// ```rust,ignore
    /// if let Some(uptime) = pipeline.uptime() {
    ///     println!("Pipeline running for {:?}", uptime);
    /// }
    /// ```
    #[must_use]
    pub fn uptime(&self) -> Option<Duration> {
        self.initialization_time.map(|t| t.elapsed())
    }

    /// Get the initialization status.
    ///
    /// Returns `true` if the pipeline has completed initialization
    /// (either via `create_and_warm()` or `warm()`).
    ///
    /// Note: This is different from [`is_ready()`](Self::is_ready), which also
    /// checks current health status.
    #[must_use]
    pub fn is_initialized(&self) -> bool {
        self.initialized.load(Ordering::SeqCst)
    }

    /// Get a status line for quick monitoring.
    ///
    /// Returns a concise string like:
    /// `WARM: 13/13 models | 24.0GB/24.0GB VRAM | OK`
    ///
    /// # Example
    ///
    /// ```rust,ignore
    /// println!("{}", pipeline.status_line());
    /// ```
    #[must_use]
    pub fn status_line(&self) -> String {
        WarmDiagnostics::status_line(&self.loader)
    }

    /// Handle a fatal error by logging diagnostics and exiting.
    ///
    /// This is a static method used during initialization before
    /// we have a full pipeline instance.
    fn handle_fatal_error_static(error: &WarmError) -> ! {
        tracing::error!(
            exit_code = error.exit_code(),
            category = %error.category(),
            error_code = %error.error_code(),
            "FATAL: WarmEmbeddingPipeline initialization failed"
        );
        tracing::error!("Error details: {}", error);

        std::process::exit(error.exit_code())
    }
}

// ============================================================================
// Unit Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use crate::warm::handle::ModelHandle;
    use crate::warm::health::WarmHealthStatus;
    use crate::warm::registry::{EMBEDDING_MODEL_IDS, FUSEMOE_MODEL_ID, TOTAL_MODEL_COUNT};

    /// Create a test config that doesn't require real files.
    fn test_config() -> WarmConfig {
        let mut config = WarmConfig::default();
        config.enable_test_inference = true;
        config
    }

    /// Helper to create a test handle
    fn test_handle() -> ModelHandle {
        ModelHandle::new(0x1000_0000, 512 * 1024 * 1024, 0, 0xDEAD_BEEF)
    }

    // ========================================================================
    // Test 1: Pipeline Creation Without Warming
    // ========================================================================

    #[test]
    fn test_pipeline_creation() {
        let config = test_config();
        let pipeline = WarmEmbeddingPipeline::new(config).expect("Failed to create pipeline");

        // Pipeline should exist but not be ready
        assert!(!pipeline.is_ready());
        assert!(!pipeline.is_initialized());
        assert!(pipeline.uptime().is_none());
    }

    // ========================================================================
    // Test 2: Pipeline Warm All Models
    // ========================================================================

    #[test]
    fn test_pipeline_warm_all_models() {
        let config = test_config();
        let mut pipeline = WarmEmbeddingPipeline::new(config).expect("Failed to create pipeline");

        // Manually transition all models to Warm state (simulating load)
        {
            let mut registry = pipeline.registry().write().unwrap();
            for model_id in EMBEDDING_MODEL_IDS.iter().chain(std::iter::once(&FUSEMOE_MODEL_ID)) {
                registry.start_loading(model_id).unwrap();
                registry.mark_validating(model_id).unwrap();
                registry.mark_warm(model_id, test_handle()).unwrap();
            }
        }

        // Manually set initialized flag (since we bypassed normal loading)
        pipeline.initialized.store(true, Ordering::SeqCst);
        pipeline.initialization_time = Some(Instant::now());

        // Recreate health checker to pick up new state
        pipeline.health_checker = WarmHealthChecker::from_loader(&pipeline.loader);

        // Now pipeline should be ready
        assert!(pipeline.is_ready());
        assert!(pipeline.is_initialized());
    }

    // ========================================================================
    // Test 3: Pipeline Is Ready Check
    // ========================================================================

    #[test]
    fn test_pipeline_is_ready() {
        let config = test_config();
        let pipeline = WarmEmbeddingPipeline::new(config).expect("Failed to create pipeline");

        // Initially not ready
        assert!(!pipeline.is_ready());

        // After marking initialized but with no warm models, still not ready
        // because health check will fail
        pipeline.initialized.store(true, Ordering::SeqCst);
        assert!(!pipeline.is_ready()); // Health checker reports not healthy
    }

    // ========================================================================
    // Test 4: Pipeline Health Check
    // ========================================================================

    #[test]
    fn test_pipeline_health() {
        let config = test_config();
        let pipeline = WarmEmbeddingPipeline::new(config).expect("Failed to create pipeline");

        let health = pipeline.health();

        // Initial state should be Loading (models pending)
        assert_eq!(health.status, WarmHealthStatus::Loading);
        assert_eq!(health.models_total, TOTAL_MODEL_COUNT);
        assert_eq!(health.models_warm, 0);
        assert_eq!(health.models_pending, TOTAL_MODEL_COUNT);
        assert_eq!(health.models_failed, 0);
        assert!(health.error_messages.is_empty());
    }

    // ========================================================================
    // Test 5: Pipeline Diagnostics
    // ========================================================================

    #[test]
    fn test_pipeline_diagnostics() {
        let config = test_config();
        let pipeline = WarmEmbeddingPipeline::new(config).expect("Failed to create pipeline");

        let report = pipeline.diagnostics();

        // Verify report structure
        assert!(!report.timestamp.is_empty());
        assert!(!report.system.hostname.is_empty() || report.system.hostname == "unknown");
        assert_eq!(report.models.len(), TOTAL_MODEL_COUNT);
        assert_eq!(report.warm_count(), 0);
        assert_eq!(report.failed_count(), 0);
    }

    // ========================================================================
    // Test 6: Pipeline Uptime Tracking
    // ========================================================================

    #[test]
    fn test_pipeline_uptime() {
        let config = test_config();
        let mut pipeline = WarmEmbeddingPipeline::new(config).expect("Failed to create pipeline");

        // Before initialization, no uptime
        assert!(pipeline.uptime().is_none());

        // Simulate initialization
        pipeline.initialization_time = Some(Instant::now());

        // After initialization, uptime should be tracked
        let uptime = pipeline.uptime();
        assert!(uptime.is_some());
        // Just verify we can access the uptime value - Duration is inherently non-negative
        let _ = uptime.unwrap();
    }

    // ========================================================================
    // Test 7: Pipeline Registry Access
    // ========================================================================

    #[test]
    fn test_pipeline_registry_access() {
        let config = test_config();
        let pipeline = WarmEmbeddingPipeline::new(config).expect("Failed to create pipeline");

        let registry = pipeline.registry();

        // Verify we can read the registry
        let guard = registry.read().unwrap();
        assert_eq!(guard.model_count(), TOTAL_MODEL_COUNT);

        // Verify all models are registered
        for model_id in EMBEDDING_MODEL_IDS {
            assert!(guard.get_state(model_id).is_some());
        }
        assert!(guard.get_state(FUSEMOE_MODEL_ID).is_some());
    }

    // ========================================================================
    // Test 8: Combined Create and Warm (Simulated)
    // ========================================================================

    #[test]
    fn test_create_and_warm() {
        // Note: We can't fully test create_and_warm() because it calls exit()
        // on failure. Instead, we test the happy path components.

        let config = test_config();
        let mut pipeline = WarmEmbeddingPipeline::new(config).expect("Failed to create pipeline");

        // Simulate successful loading
        {
            let mut registry = pipeline.registry().write().unwrap();
            for model_id in EMBEDDING_MODEL_IDS.iter().chain(std::iter::once(&FUSEMOE_MODEL_ID)) {
                registry.start_loading(model_id).unwrap();
                registry.mark_validating(model_id).unwrap();
                registry.mark_warm(model_id, test_handle()).unwrap();
            }
        }

        // Complete initialization
        pipeline.initialized.store(true, Ordering::SeqCst);
        pipeline.initialization_time = Some(Instant::now());
        pipeline.health_checker = WarmHealthChecker::from_loader(&pipeline.loader);

        // Verify all conditions that create_and_warm would check
        assert!(pipeline.is_ready());
        assert!(pipeline.is_initialized());

        let health = pipeline.health();
        assert_eq!(health.status, WarmHealthStatus::Healthy);
        assert_eq!(health.models_warm, TOTAL_MODEL_COUNT);
    }

    // ========================================================================
    // Additional Tests
    // ========================================================================

    #[test]
    fn test_pipeline_status_line() {
        let config = test_config();
        let pipeline = WarmEmbeddingPipeline::new(config).expect("Failed to create pipeline");

        let status = pipeline.status_line();

        // Verify format
        assert!(status.contains("WARM:"));
        assert!(status.contains("models"));
        assert!(status.contains("VRAM"));
        // Initial state should show LOADING
        assert!(status.contains("LOADING:") || status.contains("0/13"));
    }

    #[test]
    fn test_pipeline_loader_access() {
        let config = test_config();
        let pipeline = WarmEmbeddingPipeline::new(config).expect("Failed to create pipeline");

        let loader = pipeline.loader();
        let summary = loader.loading_summary();

        assert_eq!(summary.total_models, TOTAL_MODEL_COUNT);
        assert!(!summary.all_warm());
    }

    #[test]
    fn test_pipeline_is_initialized_vs_is_ready() {
        let config = test_config();
        let pipeline = WarmEmbeddingPipeline::new(config).expect("Failed to create pipeline");

        // Initially neither initialized nor ready
        assert!(!pipeline.is_initialized());
        assert!(!pipeline.is_ready());

        // After setting initialized flag, still not ready (health check fails)
        pipeline.initialized.store(true, Ordering::SeqCst);
        assert!(pipeline.is_initialized());
        assert!(!pipeline.is_ready()); // Health check still fails
    }

    #[test]
    fn test_pipeline_warm_already_initialized() {
        let config = test_config();
        let mut pipeline = WarmEmbeddingPipeline::new(config).expect("Failed to create pipeline");

        // Simulate initialization
        pipeline.initialized.store(true, Ordering::SeqCst);

        // Second warm() should be a no-op
        let result = pipeline.warm();
        assert!(result.is_ok());
    }

    #[test]
    fn test_pipeline_health_check_unhealthy() {
        let config = test_config();
        let pipeline = WarmEmbeddingPipeline::new(config).expect("Failed to create pipeline");

        // Fail one model
        {
            let mut registry = pipeline.registry().write().unwrap();
            registry.start_loading("E1_Semantic").unwrap();
            registry
                .mark_failed("E1_Semantic", 102, "CUDA allocation failed")
                .unwrap();
        }

        let health = pipeline.health();
        assert_eq!(health.status, WarmHealthStatus::Unhealthy);
        assert_eq!(health.models_failed, 1);
        assert!(!health.error_messages.is_empty());
        assert!(health.error_messages[0].contains("E1_Semantic"));
    }

    #[test]
    fn test_pipeline_uptime_increases() {
        let config = test_config();
        let mut pipeline = WarmEmbeddingPipeline::new(config).expect("Failed to create pipeline");

        // Set initialization time
        pipeline.initialization_time = Some(Instant::now());

        let uptime1 = pipeline.uptime().unwrap();

        // Wait a tiny bit
        std::thread::sleep(std::time::Duration::from_millis(1));

        let uptime2 = pipeline.uptime().unwrap();
        assert!(uptime2 > uptime1);
    }

    #[test]
    fn test_pipeline_diagnostics_with_failed_models() {
        let config = test_config();
        let pipeline = WarmEmbeddingPipeline::new(config).expect("Failed to create pipeline");

        // Fail some models
        {
            let mut registry = pipeline.registry().write().unwrap();
            registry.start_loading("E1_Semantic").unwrap();
            registry
                .mark_failed("E1_Semantic", 102, "Error 1")
                .unwrap();

            registry.start_loading("E2_TemporalRecent").unwrap();
            registry
                .mark_failed("E2_TemporalRecent", 104, "Error 2")
                .unwrap();
        }

        let report = pipeline.diagnostics();
        assert_eq!(report.failed_count(), 2);
        assert!(report.has_errors());
        assert_eq!(report.errors.len(), 2);
    }

    #[test]
    fn test_pipeline_concurrent_health_checks() {
        let config = test_config();
        let pipeline = WarmEmbeddingPipeline::new(config).expect("Failed to create pipeline");

        // Simulate concurrent health checks (should not deadlock)
        let handles: Vec<_> = (0..4)
            .map(|_| {
                let health_checker = pipeline.health_checker.clone();
                std::thread::spawn(move || {
                    for _ in 0..10 {
                        let _ = health_checker.check();
                    }
                })
            })
            .collect();

        for handle in handles {
            handle.join().expect("Thread panicked");
        }
    }

    #[test]
    fn test_pipeline_memory_metrics_in_health() {
        let config = test_config();
        let pipeline = WarmEmbeddingPipeline::new(config).expect("Failed to create pipeline");

        let health = pipeline.health();

        // VRAM metrics should be populated
        assert!(health.vram_total_bytes() > 0);
        assert!(health.working_memory_available_bytes > 0);

        // Initially no VRAM allocated
        assert_eq!(health.vram_allocated_bytes, 0);
    }
}
