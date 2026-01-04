//! Health Check API for Warm Model Loading
//!
//! Provides real-time health status reporting for the warm loading system.
//! This module is critical for monitoring, orchestration, and debugging.
//!
//! # Design Principles
//!
//! - **NO WORKAROUNDS OR FALLBACKS**: Health checks report the true system state
//! - **NO MOCK DATA**: All status information comes from actual components
//! - **THREAD SAFE**: Safe for concurrent access from multiple monitoring threads
//!
//! # Health States
//!
//! | Status | Condition |
//! |--------|-----------|
//! | `Healthy` | All registered models are in `Warm` state |
//! | `Loading` | At least one model is `Loading` or `Validating`, none `Failed` |
//! | `Unhealthy` | At least one model is in `Failed` state |
//! | `NotInitialized` | No models registered or registry unavailable |
//!
//! # Example
//!
//! ```rust,ignore
//! use context_graph_embeddings::warm::health::{WarmHealthChecker, WarmHealthStatus};
//! use context_graph_embeddings::warm::WarmLoader;
//!
//! let loader = WarmLoader::new(config)?;
//! let checker = WarmHealthChecker::from_loader(&loader);
//!
//! // Quick status check
//! if checker.is_healthy() {
//!     println!("All models ready for inference");
//! }
//!
//! // Detailed health check
//! let health = checker.check();
//! println!("Status: {:?}, Models warm: {}/{}",
//!     health.status, health.models_warm, health.models_total);
//! ```
//!
//! # Requirements Implemented
//!
//! - REQ-WARM-006: Health check status reporting
//! - REQ-WARM-007: Per-model state visibility

use std::time::{Duration, Instant};

use super::loader::WarmLoader;
use super::memory_pool::WarmMemoryPools;
use super::registry::SharedWarmRegistry;
use super::state::WarmModelState;

// ============================================================================
// Health Status Enum
// ============================================================================

/// Overall health status of the warm loading system.
///
/// Represents the aggregate state of all models in the system.
/// Used for quick health checks and monitoring integrations.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum WarmHealthStatus {
    /// All registered models are in `Warm` state and ready for inference.
    ///
    /// This is the only status that indicates the system is fully operational.
    Healthy,

    /// At least one model is still loading or validating.
    ///
    /// No models have failed. The system is progressing toward `Healthy`.
    Loading,

    /// At least one model has failed to load or validate.
    ///
    /// The system cannot serve inference requests. Check `error_messages`
    /// in [`WarmHealthCheck`] for details.
    Unhealthy,

    /// The registry is empty or inaccessible.
    ///
    /// This occurs when:
    /// - No models have been registered yet
    /// - The registry lock is poisoned (thread panic)
    /// - The system has not been initialized
    NotInitialized,
}

impl WarmHealthStatus {
    /// Returns `true` if the system is ready for inference.
    #[inline]
    #[must_use]
    pub fn is_healthy(&self) -> bool {
        matches!(self, Self::Healthy)
    }

    /// Returns `true` if models are still loading.
    #[inline]
    #[must_use]
    pub fn is_loading(&self) -> bool {
        matches!(self, Self::Loading)
    }

    /// Returns `true` if at least one model has failed.
    #[inline]
    #[must_use]
    pub fn is_unhealthy(&self) -> bool {
        matches!(self, Self::Unhealthy)
    }

    /// Returns `true` if the system is not initialized.
    #[inline]
    #[must_use]
    pub fn is_not_initialized(&self) -> bool {
        matches!(self, Self::NotInitialized)
    }

    /// Get a human-readable status string.
    #[must_use]
    pub fn as_str(&self) -> &'static str {
        match self {
            Self::Healthy => "healthy",
            Self::Loading => "loading",
            Self::Unhealthy => "unhealthy",
            Self::NotInitialized => "not_initialized",
        }
    }
}

impl std::fmt::Display for WarmHealthStatus {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.as_str())
    }
}

// ============================================================================
// Health Check Result
// ============================================================================

/// Detailed health check result.
///
/// Contains comprehensive information about the warm loading system state,
/// including model counts, VRAM usage, and error details.
#[derive(Debug, Clone)]
pub struct WarmHealthCheck {
    /// Overall health status.
    pub status: WarmHealthStatus,

    /// Total number of registered models.
    pub models_total: usize,

    /// Number of models in `Warm` state.
    pub models_warm: usize,

    /// Number of models in `Loading` or `Validating` state.
    pub models_loading: usize,

    /// Number of models in `Failed` state.
    pub models_failed: usize,

    /// Number of models in `Pending` state.
    pub models_pending: usize,

    /// Total VRAM allocated for models (bytes).
    pub vram_allocated_bytes: usize,

    /// Available VRAM for model allocations (bytes).
    pub vram_available_bytes: usize,

    /// Working memory allocated (bytes).
    pub working_memory_allocated_bytes: usize,

    /// Working memory available (bytes).
    pub working_memory_available_bytes: usize,

    /// System uptime since checker creation.
    pub uptime: Option<Duration>,

    /// Timestamp of this health check.
    pub last_check: Instant,

    /// Error messages from failed models.
    ///
    /// Format: `["model_id: error message", ...]`
    pub error_messages: Vec<String>,
}

impl WarmHealthCheck {
    /// Create an empty health check result for uninitialized state.
    #[must_use]
    pub fn not_initialized() -> Self {
        Self {
            status: WarmHealthStatus::NotInitialized,
            models_total: 0,
            models_warm: 0,
            models_loading: 0,
            models_failed: 0,
            models_pending: 0,
            vram_allocated_bytes: 0,
            vram_available_bytes: 0,
            working_memory_allocated_bytes: 0,
            working_memory_available_bytes: 0,
            uptime: None,
            last_check: Instant::now(),
            error_messages: Vec::new(),
        }
    }

    /// Check if the system is healthy.
    #[inline]
    #[must_use]
    pub fn is_healthy(&self) -> bool {
        self.status.is_healthy()
    }

    /// Get warm model percentage (0.0 - 100.0).
    #[must_use]
    pub fn warm_percentage(&self) -> f64 {
        if self.models_total == 0 {
            return 0.0;
        }
        (self.models_warm as f64 / self.models_total as f64) * 100.0
    }

    /// Get total VRAM capacity (allocated + available).
    #[must_use]
    pub fn vram_total_bytes(&self) -> usize {
        self.vram_allocated_bytes
            .saturating_add(self.vram_available_bytes)
    }

    /// Get VRAM utilization percentage (0.0 - 1.0).
    #[must_use]
    pub fn vram_utilization(&self) -> f64 {
        let total = self.vram_total_bytes();
        if total == 0 {
            return 0.0;
        }
        self.vram_allocated_bytes as f64 / total as f64
    }
}

impl Default for WarmHealthCheck {
    fn default() -> Self {
        Self::not_initialized()
    }
}

// ============================================================================
// Health Checker Service
// ============================================================================

/// Health check service for the warm loading system.
///
/// Provides methods to query the current health status of all loaded models.
/// Thread-safe for concurrent access from monitoring systems.
///
/// # Example
///
/// ```rust,ignore
/// let checker = WarmHealthChecker::from_loader(&loader);
///
/// // Quick status check (no allocations)
/// match checker.status() {
///     WarmHealthStatus::Healthy => serve_requests(),
///     WarmHealthStatus::Loading => wait_for_startup(),
///     WarmHealthStatus::Unhealthy => alert_oncall(),
///     WarmHealthStatus::NotInitialized => panic!("System not initialized"),
/// }
///
/// // Detailed check with metrics
/// let health = checker.check();
/// prometheus::gauge!("models_warm").set(health.models_warm as f64);
/// ```
pub struct WarmHealthChecker {
    /// Shared registry for model state access.
    registry: SharedWarmRegistry,

    /// Memory pools for VRAM usage queries.
    memory_pools: WarmMemoryPools,

    /// Start time for uptime tracking.
    start_time: Instant,
}

impl WarmHealthChecker {
    /// Create a health checker from a [`WarmLoader`].
    ///
    /// Extracts the registry and memory pools from the loader.
    /// The checker maintains references to these components for querying.
    ///
    /// # Arguments
    ///
    /// * `loader` - The warm loader to monitor
    ///
    /// # Example
    ///
    /// ```rust,ignore
    /// let loader = WarmLoader::new(config)?;
    /// let checker = WarmHealthChecker::from_loader(&loader);
    /// ```
    #[must_use]
    pub fn from_loader(loader: &WarmLoader) -> Self {
        Self {
            registry: loader.registry().clone(),
            memory_pools: loader.memory_pools().clone(),
            start_time: Instant::now(),
        }
    }

    /// Create a health checker directly from components.
    ///
    /// Useful for testing or when constructing the checker independently.
    ///
    /// # Arguments
    ///
    /// * `registry` - Shared registry for model state access
    /// * `memory_pools` - Memory pools for VRAM queries
    ///
    /// # Example
    ///
    /// ```rust,ignore
    /// let registry = Arc::new(RwLock::new(WarmModelRegistry::new()));
    /// let pools = WarmMemoryPools::rtx_5090();
    /// let checker = WarmHealthChecker::new(registry, pools);
    /// ```
    #[must_use]
    pub fn new(registry: SharedWarmRegistry, memory_pools: WarmMemoryPools) -> Self {
        Self {
            registry,
            memory_pools,
            start_time: Instant::now(),
        }
    }

    /// Perform a detailed health check.
    ///
    /// Queries the registry for model states and memory pools for VRAM usage.
    /// Returns a comprehensive [`WarmHealthCheck`] result.
    ///
    /// # Thread Safety
    ///
    /// Acquires a read lock on the registry. Safe for concurrent calls.
    /// Returns `NotInitialized` status if the lock is poisoned.
    ///
    /// # Returns
    ///
    /// A [`WarmHealthCheck`] containing status, model counts, and VRAM metrics.
    #[must_use]
    pub fn check(&self) -> WarmHealthCheck {
        let last_check = Instant::now();
        let uptime = Some(self.start_time.elapsed());

        // Attempt to read the registry
        let registry = match self.registry.read() {
            Ok(r) => r,
            Err(_) => {
                // Lock poisoned - return not initialized
                return WarmHealthCheck {
                    status: WarmHealthStatus::NotInitialized,
                    uptime,
                    last_check,
                    ..WarmHealthCheck::not_initialized()
                };
            }
        };

        // Check if any models are registered
        let models_total = registry.model_count();
        if models_total == 0 {
            return WarmHealthCheck {
                status: WarmHealthStatus::NotInitialized,
                models_total: 0,
                uptime,
                last_check,
                ..WarmHealthCheck::not_initialized()
            };
        }

        // Count models in each state
        let mut models_warm = 0usize;
        let mut models_loading = 0usize;
        let mut models_failed = 0usize;
        let mut models_pending = 0usize;
        let mut error_messages = Vec::new();

        // Iterate through all model states
        for entry in registry.loading_order() {
            if let Some(state) = registry.get_state(&entry) {
                match &state {
                    WarmModelState::Warm => models_warm += 1,
                    WarmModelState::Loading { .. } | WarmModelState::Validating => {
                        models_loading += 1;
                    }
                    WarmModelState::Failed { error_code, error_message } => {
                        models_failed += 1;
                        error_messages.push(format!(
                            "{}: [{}] {}",
                            entry, error_code, error_message
                        ));
                    }
                    WarmModelState::Pending => models_pending += 1,
                }
            }
        }

        // Determine overall status
        let status = if models_failed > 0 {
            WarmHealthStatus::Unhealthy
        } else if models_loading > 0 || models_pending > 0 {
            // If any model is still pending or loading, we're in loading state
            if models_warm == 0 && models_loading == 0 {
                // All pending, nothing started
                WarmHealthStatus::Loading
            } else {
                WarmHealthStatus::Loading
            }
        } else if models_warm == models_total {
            WarmHealthStatus::Healthy
        } else {
            // Unexpected state - should not happen
            WarmHealthStatus::NotInitialized
        };

        // Get VRAM metrics from memory pools
        let model_allocations = self.memory_pools.list_model_allocations();
        let vram_allocated_bytes: usize = model_allocations.iter().map(|a| a.size_bytes).sum();
        let vram_available_bytes = self.memory_pools.available_model_bytes();

        // Get working memory metrics
        let working_memory_available_bytes = self.memory_pools.available_working_bytes();
        let working_memory_total = self.memory_pools.working_pool_capacity();
        let working_memory_allocated_bytes =
            working_memory_total.saturating_sub(working_memory_available_bytes);

        WarmHealthCheck {
            status,
            models_total,
            models_warm,
            models_loading,
            models_failed,
            models_pending,
            vram_allocated_bytes,
            vram_available_bytes,
            working_memory_allocated_bytes,
            working_memory_available_bytes,
            uptime,
            last_check,
            error_messages,
        }
    }

    /// Quick status check without detailed metrics.
    ///
    /// More efficient than [`check()`](Self::check) when only the status is needed.
    ///
    /// # Returns
    ///
    /// The current [`WarmHealthStatus`].
    #[must_use]
    pub fn status(&self) -> WarmHealthStatus {
        // Attempt to read the registry
        let registry = match self.registry.read() {
            Ok(r) => r,
            Err(_) => return WarmHealthStatus::NotInitialized,
        };

        // Check if any models are registered
        if registry.model_count() == 0 {
            return WarmHealthStatus::NotInitialized;
        }

        // Quick check: any failed?
        if registry.any_failed() {
            return WarmHealthStatus::Unhealthy;
        }

        // Quick check: all warm?
        if registry.all_warm() {
            return WarmHealthStatus::Healthy;
        }

        // Otherwise, still loading
        WarmHealthStatus::Loading
    }

    /// Check if the system is healthy.
    ///
    /// Convenience method equivalent to `self.status().is_healthy()`.
    ///
    /// # Returns
    ///
    /// `true` if all models are in `Warm` state.
    #[inline]
    #[must_use]
    pub fn is_healthy(&self) -> bool {
        self.status().is_healthy()
    }

    /// Get the uptime since the checker was created.
    ///
    /// # Returns
    ///
    /// Duration since the health checker was instantiated.
    #[must_use]
    pub fn uptime(&self) -> Duration {
        self.start_time.elapsed()
    }

    /// Get a reference to the registry.
    ///
    /// For advanced use cases requiring direct registry access.
    #[must_use]
    pub fn registry(&self) -> &SharedWarmRegistry {
        &self.registry
    }

    /// Get a reference to the memory pools.
    ///
    /// For advanced use cases requiring direct memory pool access.
    #[must_use]
    pub fn memory_pools(&self) -> &WarmMemoryPools {
        &self.memory_pools
    }
}

impl Clone for WarmHealthChecker {
    fn clone(&self) -> Self {
        Self {
            registry: self.registry.clone(),
            memory_pools: self.memory_pools.clone(),
            start_time: self.start_time,
        }
    }
}

// ============================================================================
// Unit Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use crate::warm::config::WarmConfig;
    use crate::warm::handle::ModelHandle;
    use crate::warm::registry::WarmModelRegistry;
    use std::sync::{Arc, RwLock};
    use std::thread;

    /// Helper to create a test config
    fn test_config() -> WarmConfig {
        WarmConfig::default()
    }

    /// Helper to create a test handle
    fn test_handle() -> ModelHandle {
        ModelHandle::new(0x1000_0000, 512 * 1024 * 1024, 0, 0xDEAD_BEEF)
    }

    // ========================================================================
    // Test 1: Health Status Enum Variants
    // ========================================================================

    #[test]
    fn test_health_status_enum_variants() {
        // Verify all variants exist and have correct predicates
        let healthy = WarmHealthStatus::Healthy;
        assert!(healthy.is_healthy());
        assert!(!healthy.is_loading());
        assert!(!healthy.is_unhealthy());
        assert!(!healthy.is_not_initialized());
        assert_eq!(healthy.as_str(), "healthy");

        let loading = WarmHealthStatus::Loading;
        assert!(!loading.is_healthy());
        assert!(loading.is_loading());
        assert!(!loading.is_unhealthy());
        assert!(!loading.is_not_initialized());
        assert_eq!(loading.as_str(), "loading");

        let unhealthy = WarmHealthStatus::Unhealthy;
        assert!(!unhealthy.is_healthy());
        assert!(!unhealthy.is_loading());
        assert!(unhealthy.is_unhealthy());
        assert!(!unhealthy.is_not_initialized());
        assert_eq!(unhealthy.as_str(), "unhealthy");

        let not_init = WarmHealthStatus::NotInitialized;
        assert!(!not_init.is_healthy());
        assert!(!not_init.is_loading());
        assert!(!not_init.is_unhealthy());
        assert!(not_init.is_not_initialized());
        assert_eq!(not_init.as_str(), "not_initialized");
    }

    // ========================================================================
    // Test 2: Health Check Initial Not Initialized
    // ========================================================================

    #[test]
    fn test_health_check_initial_not_initialized() {
        // Create an empty registry
        let registry: SharedWarmRegistry = Arc::new(RwLock::new(WarmModelRegistry::new()));
        let pools = WarmMemoryPools::new(test_config());

        let checker = WarmHealthChecker::new(registry, pools);
        let health = checker.check();

        // Empty registry should report NotInitialized
        assert_eq!(health.status, WarmHealthStatus::NotInitialized);
        assert_eq!(health.models_total, 0);
        assert_eq!(health.models_warm, 0);
        assert_eq!(health.models_loading, 0);
        assert_eq!(health.models_failed, 0);
        assert!(health.error_messages.is_empty());

        // Quick status should also be NotInitialized
        assert_eq!(checker.status(), WarmHealthStatus::NotInitialized);
        assert!(!checker.is_healthy());
    }

    // ========================================================================
    // Test 3: Health Check Loading State
    // ========================================================================

    #[test]
    fn test_health_check_loading_state() {
        let mut registry_inner = WarmModelRegistry::new();

        // Register some models
        registry_inner
            .register_model("E1_Semantic", 500 * 1024 * 1024, 768)
            .unwrap();
        registry_inner
            .register_model("E2_TemporalRecent", 400 * 1024 * 1024, 768)
            .unwrap();
        registry_inner
            .register_model("E3_TemporalPeriodic", 400 * 1024 * 1024, 768)
            .unwrap();

        // Start loading one model
        registry_inner.start_loading("E1_Semantic").unwrap();

        // Warm another model
        registry_inner.start_loading("E2_TemporalRecent").unwrap();
        registry_inner.mark_validating("E2_TemporalRecent").unwrap();
        registry_inner
            .mark_warm("E2_TemporalRecent", test_handle())
            .unwrap();

        // Third model still pending

        let registry: SharedWarmRegistry = Arc::new(RwLock::new(registry_inner));
        let pools = WarmMemoryPools::new(test_config());

        let checker = WarmHealthChecker::new(registry, pools);
        let health = checker.check();

        // Should be Loading (some models not warm, none failed)
        assert_eq!(health.status, WarmHealthStatus::Loading);
        assert_eq!(health.models_total, 3);
        assert_eq!(health.models_warm, 1);
        assert_eq!(health.models_loading, 1); // E1 is loading
        assert_eq!(health.models_pending, 1); // E3 is pending
        assert_eq!(health.models_failed, 0);
        assert!(health.error_messages.is_empty());
    }

    // ========================================================================
    // Test 4: Health Check Healthy State
    // ========================================================================

    #[test]
    fn test_health_check_healthy_state() {
        let mut registry_inner = WarmModelRegistry::new();

        // Register and warm all models
        let models = ["E1_Semantic", "E2_TemporalRecent", "E3_TemporalPeriodic"];
        for model_id in models {
            registry_inner
                .register_model(model_id, 500 * 1024 * 1024, 768)
                .unwrap();
            registry_inner.start_loading(model_id).unwrap();
            registry_inner.mark_validating(model_id).unwrap();
            registry_inner.mark_warm(model_id, test_handle()).unwrap();
        }

        let registry: SharedWarmRegistry = Arc::new(RwLock::new(registry_inner));
        let pools = WarmMemoryPools::new(test_config());

        let checker = WarmHealthChecker::new(registry, pools);
        let health = checker.check();

        // Should be Healthy (all models warm)
        assert_eq!(health.status, WarmHealthStatus::Healthy);
        assert_eq!(health.models_total, 3);
        assert_eq!(health.models_warm, 3);
        assert_eq!(health.models_loading, 0);
        assert_eq!(health.models_failed, 0);
        assert!(health.error_messages.is_empty());

        // Quick checks
        assert!(checker.is_healthy());
        assert_eq!(checker.status(), WarmHealthStatus::Healthy);
        assert!((health.warm_percentage() - 100.0).abs() < 0.01);
    }

    // ========================================================================
    // Test 5: Health Check Unhealthy State
    // ========================================================================

    #[test]
    fn test_health_check_unhealthy_state() {
        let mut registry_inner = WarmModelRegistry::new();

        // Register models
        registry_inner
            .register_model("E1_Semantic", 500 * 1024 * 1024, 768)
            .unwrap();
        registry_inner
            .register_model("E2_TemporalRecent", 400 * 1024 * 1024, 768)
            .unwrap();

        // Warm one model
        registry_inner.start_loading("E1_Semantic").unwrap();
        registry_inner.mark_validating("E1_Semantic").unwrap();
        registry_inner
            .mark_warm("E1_Semantic", test_handle())
            .unwrap();

        // Fail another model
        registry_inner.start_loading("E2_TemporalRecent").unwrap();
        registry_inner
            .mark_failed("E2_TemporalRecent", 102, "CUDA allocation failed")
            .unwrap();

        let registry: SharedWarmRegistry = Arc::new(RwLock::new(registry_inner));
        let pools = WarmMemoryPools::new(test_config());

        let checker = WarmHealthChecker::new(registry, pools);
        let health = checker.check();

        // Should be Unhealthy (at least one failed)
        assert_eq!(health.status, WarmHealthStatus::Unhealthy);
        assert_eq!(health.models_total, 2);
        assert_eq!(health.models_warm, 1);
        assert_eq!(health.models_failed, 1);
        assert_eq!(health.error_messages.len(), 1);
        assert!(health.error_messages[0].contains("E2_TemporalRecent"));
        assert!(health.error_messages[0].contains("CUDA allocation failed"));

        // Quick checks
        assert!(!checker.is_healthy());
        assert_eq!(checker.status(), WarmHealthStatus::Unhealthy);
    }

    // ========================================================================
    // Test 6: Health Checker From Loader
    // ========================================================================

    #[test]
    fn test_health_checker_from_loader() {
        let config = test_config();
        let loader = WarmLoader::new(config).expect("Failed to create loader");

        let checker = WarmHealthChecker::from_loader(&loader);

        // Loader starts with models in Pending state
        let health = checker.check();

        // Should be Loading (models registered but pending)
        assert_eq!(health.status, WarmHealthStatus::Loading);
        assert!(health.models_total > 0);
        assert_eq!(health.models_warm, 0);
        assert!(health.models_pending > 0);

        // Verify uptime is tracked
        assert!(health.uptime.is_some());
        assert!(checker.uptime() >= Duration::ZERO);
    }

    // ========================================================================
    // Test 7: Uptime Tracking
    // ========================================================================

    #[test]
    fn test_uptime_tracking() {
        let registry: SharedWarmRegistry = Arc::new(RwLock::new(WarmModelRegistry::new()));
        let pools = WarmMemoryPools::new(test_config());

        let checker = WarmHealthChecker::new(registry, pools);

        // Initial uptime should be near zero
        let uptime1 = checker.uptime();
        assert!(uptime1.as_millis() < 100);

        // Wait a bit
        thread::sleep(Duration::from_millis(10));

        // Uptime should increase
        let uptime2 = checker.uptime();
        assert!(uptime2 > uptime1);

        // Health check should include uptime
        let health = checker.check();
        assert!(health.uptime.is_some());
        assert!(health.uptime.unwrap() >= uptime1);
    }

    // ========================================================================
    // Test 8: Error Messages Populated
    // ========================================================================

    #[test]
    fn test_error_messages_populated() {
        let mut registry_inner = WarmModelRegistry::new();

        // Register and fail multiple models with different errors
        registry_inner
            .register_model("E1_Semantic", 500 * 1024 * 1024, 768)
            .unwrap();
        registry_inner
            .register_model("E2_TemporalRecent", 400 * 1024 * 1024, 768)
            .unwrap();
        registry_inner
            .register_model("E3_TemporalPeriodic", 400 * 1024 * 1024, 768)
            .unwrap();

        // Fail E1 with one error
        registry_inner.start_loading("E1_Semantic").unwrap();
        registry_inner
            .mark_failed("E1_Semantic", 104, "Insufficient VRAM")
            .unwrap();

        // Fail E2 with different error
        registry_inner.start_loading("E2_TemporalRecent").unwrap();
        registry_inner
            .mark_failed("E2_TemporalRecent", 103, "NaN in weights")
            .unwrap();

        // E3 is still pending

        let registry: SharedWarmRegistry = Arc::new(RwLock::new(registry_inner));
        let pools = WarmMemoryPools::new(test_config());

        let checker = WarmHealthChecker::new(registry, pools);
        let health = checker.check();

        // Should have 2 error messages
        assert_eq!(health.models_failed, 2);
        assert_eq!(health.error_messages.len(), 2);

        // Verify error messages contain model IDs and error info
        let all_messages = health.error_messages.join("\n");
        assert!(all_messages.contains("E1_Semantic"));
        assert!(all_messages.contains("Insufficient VRAM"));
        assert!(all_messages.contains("104"));
        assert!(all_messages.contains("E2_TemporalRecent"));
        assert!(all_messages.contains("NaN in weights"));
        assert!(all_messages.contains("103"));
    }

    // ========================================================================
    // Additional Tests
    // ========================================================================

    #[test]
    fn test_health_check_not_initialized_default() {
        let health = WarmHealthCheck::not_initialized();

        assert_eq!(health.status, WarmHealthStatus::NotInitialized);
        assert_eq!(health.models_total, 0);
        assert_eq!(health.warm_percentage(), 0.0);
        assert!(!health.is_healthy());
    }

    #[test]
    fn test_health_check_default() {
        let health = WarmHealthCheck::default();
        assert_eq!(health.status, WarmHealthStatus::NotInitialized);
    }

    #[test]
    fn test_health_status_display() {
        assert_eq!(format!("{}", WarmHealthStatus::Healthy), "healthy");
        assert_eq!(format!("{}", WarmHealthStatus::Loading), "loading");
        assert_eq!(format!("{}", WarmHealthStatus::Unhealthy), "unhealthy");
        assert_eq!(
            format!("{}", WarmHealthStatus::NotInitialized),
            "not_initialized"
        );
    }

    #[test]
    fn test_vram_metrics() {
        let mut registry_inner = WarmModelRegistry::new();
        registry_inner
            .register_model("E1_Semantic", 500 * 1024 * 1024, 768)
            .unwrap();

        let registry: SharedWarmRegistry = Arc::new(RwLock::new(registry_inner));
        let mut pools = WarmMemoryPools::new(test_config());

        // Allocate some VRAM
        pools
            .allocate_model("E1_Semantic", 500 * 1024 * 1024, 0x1000)
            .unwrap();

        let checker = WarmHealthChecker::new(registry, pools);
        let health = checker.check();

        // Verify VRAM metrics are populated
        assert_eq!(health.vram_allocated_bytes, 500 * 1024 * 1024);
        assert!(health.vram_available_bytes > 0);
        assert!(health.vram_total_bytes() > 0);
        assert!(health.vram_utilization() > 0.0);
    }

    #[test]
    fn test_working_memory_metrics() {
        // Register at least one model so check() doesn't return early
        let mut registry_inner = WarmModelRegistry::new();
        registry_inner
            .register_model("E1_Semantic", 500 * 1024 * 1024, 768)
            .unwrap();
        registry_inner.start_loading("E1_Semantic").unwrap();
        registry_inner.mark_validating("E1_Semantic").unwrap();
        registry_inner
            .mark_warm("E1_Semantic", test_handle())
            .unwrap();

        let registry: SharedWarmRegistry = Arc::new(RwLock::new(registry_inner));
        let mut pools = WarmMemoryPools::new(test_config());

        // Allocate some working memory
        pools.allocate_working(1024 * 1024 * 1024).unwrap(); // 1GB

        let checker = WarmHealthChecker::new(registry, pools);
        let health = checker.check();

        // Verify working memory metrics
        assert!(health.working_memory_allocated_bytes > 0);
        assert!(health.working_memory_available_bytes > 0);
    }

    #[test]
    fn test_checker_clone() {
        let registry: SharedWarmRegistry = Arc::new(RwLock::new(WarmModelRegistry::new()));
        let pools = WarmMemoryPools::new(test_config());

        let checker1 = WarmHealthChecker::new(registry, pools);
        let checker2 = checker1.clone();

        // Both checkers should return the same status
        assert_eq!(checker1.status(), checker2.status());
    }

    #[test]
    fn test_checker_accessors() {
        let registry: SharedWarmRegistry = Arc::new(RwLock::new(WarmModelRegistry::new()));
        let pools = WarmMemoryPools::new(test_config());

        let checker = WarmHealthChecker::new(registry.clone(), pools);

        // Verify accessors work
        assert!(std::sync::Arc::ptr_eq(checker.registry(), &registry));
        assert!(checker.memory_pools().is_within_budget());
    }

    #[test]
    fn test_validating_state_counts_as_loading() {
        let mut registry_inner = WarmModelRegistry::new();

        registry_inner
            .register_model("E1_Semantic", 500 * 1024 * 1024, 768)
            .unwrap();

        // Put model in Validating state
        registry_inner.start_loading("E1_Semantic").unwrap();
        registry_inner.mark_validating("E1_Semantic").unwrap();

        let registry: SharedWarmRegistry = Arc::new(RwLock::new(registry_inner));
        let pools = WarmMemoryPools::new(test_config());

        let checker = WarmHealthChecker::new(registry, pools);
        let health = checker.check();

        // Validating should count as loading
        assert_eq!(health.status, WarmHealthStatus::Loading);
        assert_eq!(health.models_loading, 1);
    }

    #[test]
    fn test_warm_percentage_calculation() {
        let mut registry_inner = WarmModelRegistry::new();

        // 4 models, 2 warm
        for id in ["M1", "M2", "M3", "M4"] {
            registry_inner
                .register_model(id, 100 * 1024 * 1024, 768)
                .unwrap();
        }

        // Warm 2 models
        for id in ["M1", "M2"] {
            registry_inner.start_loading(id).unwrap();
            registry_inner.mark_validating(id).unwrap();
            registry_inner.mark_warm(id, test_handle()).unwrap();
        }

        let registry: SharedWarmRegistry = Arc::new(RwLock::new(registry_inner));
        let pools = WarmMemoryPools::new(test_config());

        let checker = WarmHealthChecker::new(registry, pools);
        let health = checker.check();

        // Should be 50% warm
        assert!((health.warm_percentage() - 50.0).abs() < 0.01);
    }
}
