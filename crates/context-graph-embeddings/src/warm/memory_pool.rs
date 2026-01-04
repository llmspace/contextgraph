//! Memory pool management for warm model loading.
//!
//! # Overview
//!
//! This module provides isolated memory pools for the warm model loading system,
//! implementing a dual-pool architecture that separates model weights from
//! working memory allocations.
//!
//! # Pool Isolation Strategy
//!
//! The warm loading system uses two distinct memory pools with different
//! eviction policies:
//!
//! 1. **Model Pool (Non-Evictable)**: Stores model weights that must remain
//!    resident in VRAM for the entire application lifetime. These allocations
//!    are protected from memory pressure and CANNOT be evicted.
//!
//! 2. **Working Pool (Evictable)**: Stores temporary inference activations
//!    and intermediate tensors. These allocations CAN be reclaimed when
//!    memory pressure is detected.
//!
//! # Non-Evictable vs Evictable Semantics
//!
//! ## Non-Evictable (Model Pool)
//! - Allocations are permanent until explicitly freed
//! - Protected from CUDA memory pressure callbacks
//! - Failure to allocate is a fatal startup error (REQ-WARM-004)
//! - Must fit within the configured `vram_budget_bytes`
//!
//! ## Evictable (Working Pool)
//! - Allocations can be reclaimed under memory pressure
//! - Used for inference activations and temporary tensors
//! - Exhaustion returns `WorkingMemoryExhausted` (non-fatal)
//! - Sized by `vram_headroom_bytes` configuration
//!
//! # Thread-Safety Considerations
//!
//! The pools are designed to be used with `Arc<Mutex<WarmMemoryPools>>` for
//! thread-safe access from multiple inference workers. The internal state
//! is NOT internally synchronized; callers must provide external locking.
//!
//! Typical usage pattern:
//! ```rust,ignore
//! use std::sync::{Arc, Mutex};
//! use context_graph_embeddings::warm::memory_pool::WarmMemoryPools;
//!
//! let pools = Arc::new(Mutex::new(WarmMemoryPools::rtx_5090()));
//!
//! // Thread 1: Load model
//! {
//!     let mut guard = pools.lock().unwrap();
//!     guard.allocate_model("E1_Semantic", 800_000_000, vram_ptr)?;
//! }
//!
//! // Thread 2: Allocate working memory for inference
//! {
//!     let mut guard = pools.lock().unwrap();
//!     guard.allocate_working(50_000_000)?;
//! }
//! ```
//!
//! # Requirements Implemented
//!
//! - REQ-WARM-004: Non-evictable allocations for model weights
//! - REQ-WARM-005: Protected from memory pressure
//! - REQ-WARM-012: VRAM budget enforcement

use std::time::Instant;

use super::config::WarmConfig;
use super::error::{WarmError, WarmResult};

/// One gigabyte in bytes (used in tests).
#[cfg(test)]
const GB: usize = 1024 * 1024 * 1024;

/// Tracks a single model's VRAM allocation.
///
/// Each allocation records the model identifier, VRAM pointer, size,
/// and timestamp for diagnostics and lifecycle management.
#[derive(Debug, Clone)]
pub struct ModelAllocation {
    /// Unique model identifier (e.g., "E1_Semantic", "FuseMoE").
    pub model_id: String,
    /// Raw VRAM pointer from CUDA allocation.
    ///
    /// This is an opaque handle; the actual memory management is
    /// performed by the CUDA runtime.
    pub vram_ptr: u64,
    /// Size of allocation in bytes.
    pub size_bytes: usize,
    /// Timestamp when allocation was made.
    ///
    /// Used for diagnostics and debugging memory fragmentation.
    pub allocated_at: Instant,
}

impl ModelAllocation {
    /// Create a new model allocation record.
    #[must_use]
    pub fn new(model_id: String, vram_ptr: u64, size_bytes: usize) -> Self {
        Self {
            model_id,
            vram_ptr,
            size_bytes,
            allocated_at: Instant::now(),
        }
    }

    /// Get the age of this allocation in seconds.
    #[must_use]
    pub fn age_secs(&self) -> f64 {
        self.allocated_at.elapsed().as_secs_f64()
    }
}

/// Non-evictable memory pool for model weights.
///
/// This pool holds permanent VRAM allocations for model weights.
/// Allocations in this pool are protected from memory pressure
/// and cannot be evicted.
///
/// # Invariants
///
/// - `allocated_bytes <= capacity_bytes` (enforced by allocation methods)
/// - No duplicate `model_id` entries in `allocations`
#[derive(Debug, Clone)]
pub struct ModelMemoryPool {
    /// Total capacity in bytes (e.g., 24GB for RTX 5090).
    capacity_bytes: usize,
    /// Currently allocated bytes across all models.
    allocated_bytes: usize,
    /// Tracking information for each model allocation.
    allocations: Vec<ModelAllocation>,
}

impl ModelMemoryPool {
    /// Create a new model memory pool with the specified capacity.
    ///
    /// # Arguments
    ///
    /// * `capacity_bytes` - Maximum bytes that can be allocated for model weights
    #[must_use]
    pub fn new(capacity_bytes: usize) -> Self {
        Self {
            capacity_bytes,
            allocated_bytes: 0,
            allocations: Vec::new(),
        }
    }

    /// Get the total capacity of this pool in bytes.
    #[must_use]
    pub fn capacity(&self) -> usize {
        self.capacity_bytes
    }

    /// Get the currently allocated bytes.
    #[must_use]
    pub fn allocated(&self) -> usize {
        self.allocated_bytes
    }

    /// Get the available bytes for new allocations.
    #[must_use]
    pub fn available(&self) -> usize {
        self.capacity_bytes.saturating_sub(self.allocated_bytes)
    }

    /// Check if a model is already allocated.
    #[must_use]
    pub fn contains(&self, model_id: &str) -> bool {
        self.allocations.iter().any(|a| a.model_id == model_id)
    }

    /// Get allocation info for a specific model.
    #[must_use]
    pub fn get(&self, model_id: &str) -> Option<&ModelAllocation> {
        self.allocations.iter().find(|a| a.model_id == model_id)
    }

    /// Get all current allocations.
    #[must_use]
    pub fn allocations(&self) -> &[ModelAllocation] {
        &self.allocations
    }

    /// Allocate memory for a model.
    ///
    /// # Arguments
    ///
    /// * `model_id` - Unique identifier for the model
    /// * `size_bytes` - Size of allocation in bytes
    /// * `vram_ptr` - Raw VRAM pointer from CUDA
    ///
    /// # Errors
    ///
    /// Returns `WarmError::ModelAlreadyRegistered` if model already allocated.
    /// Returns `WarmError::VramAllocationFailed` if allocation exceeds capacity.
    pub fn allocate(
        &mut self,
        model_id: &str,
        size_bytes: usize,
        vram_ptr: u64,
    ) -> WarmResult<()> {
        // Check for duplicate allocation
        if self.contains(model_id) {
            return Err(WarmError::ModelAlreadyRegistered {
                model_id: model_id.to_string(),
            });
        }

        // Check capacity - NO FALLBACKS
        if self.allocated_bytes.saturating_add(size_bytes) > self.capacity_bytes {
            return Err(WarmError::VramAllocationFailed {
                requested_bytes: size_bytes,
                available_bytes: self.available(),
                error: format!(
                    "Model pool capacity exhausted: {} bytes requested, {} bytes available",
                    size_bytes,
                    self.available()
                ),
            });
        }

        // Record allocation
        self.allocations.push(ModelAllocation::new(
            model_id.to_string(),
            vram_ptr,
            size_bytes,
        ));
        self.allocated_bytes = self.allocated_bytes.saturating_add(size_bytes);

        Ok(())
    }

    /// Free a model allocation.
    ///
    /// # Arguments
    ///
    /// * `model_id` - Identifier of model to free
    ///
    /// # Errors
    ///
    /// Returns `WarmError::ModelNotRegistered` if model not found.
    pub fn free(&mut self, model_id: &str) -> WarmResult<usize> {
        // Find the allocation
        let idx = self
            .allocations
            .iter()
            .position(|a| a.model_id == model_id)
            .ok_or_else(|| WarmError::ModelNotRegistered {
                model_id: model_id.to_string(),
            })?;

        // Remove and update accounting
        let allocation = self.allocations.remove(idx);
        self.allocated_bytes = self.allocated_bytes.saturating_sub(allocation.size_bytes);

        Ok(allocation.size_bytes)
    }
}

/// Evictable memory pool for working memory (inference activations).
///
/// This pool holds temporary allocations for inference operations.
/// Unlike the model pool, these allocations CAN be reclaimed under
/// memory pressure.
///
/// # Invariants
///
/// - `allocated_bytes <= capacity_bytes` (enforced by allocation methods)
#[derive(Debug, Clone)]
pub struct WorkingMemoryPool {
    /// Total capacity in bytes (e.g., 8GB for RTX 5090).
    capacity_bytes: usize,
    /// Currently allocated bytes for working memory.
    allocated_bytes: usize,
}

impl WorkingMemoryPool {
    /// Create a new working memory pool with the specified capacity.
    ///
    /// # Arguments
    ///
    /// * `capacity_bytes` - Maximum bytes for working memory
    #[must_use]
    pub fn new(capacity_bytes: usize) -> Self {
        Self {
            capacity_bytes,
            allocated_bytes: 0,
        }
    }

    /// Get the total capacity of this pool in bytes.
    #[must_use]
    pub fn capacity(&self) -> usize {
        self.capacity_bytes
    }

    /// Get the currently allocated bytes.
    #[must_use]
    pub fn allocated(&self) -> usize {
        self.allocated_bytes
    }

    /// Get the available bytes for new allocations.
    #[must_use]
    pub fn available(&self) -> usize {
        self.capacity_bytes.saturating_sub(self.allocated_bytes)
    }

    /// Allocate working memory.
    ///
    /// # Arguments
    ///
    /// * `size_bytes` - Size of allocation in bytes
    ///
    /// # Errors
    ///
    /// Returns `WarmError::WorkingMemoryExhausted` if allocation exceeds capacity.
    pub fn allocate(&mut self, size_bytes: usize) -> WarmResult<()> {
        // Check capacity - NO FALLBACKS
        if self.allocated_bytes.saturating_add(size_bytes) > self.capacity_bytes {
            return Err(WarmError::WorkingMemoryExhausted {
                requested_bytes: size_bytes,
                available_bytes: self.available(),
            });
        }

        self.allocated_bytes = self.allocated_bytes.saturating_add(size_bytes);
        Ok(())
    }

    /// Free working memory.
    ///
    /// # Arguments
    ///
    /// * `size_bytes` - Size to free in bytes
    ///
    /// # Note
    ///
    /// This method silently caps the freed amount to prevent underflow.
    /// If you attempt to free more than is allocated, only the allocated
    /// amount will be freed.
    pub fn free(&mut self, size_bytes: usize) {
        self.allocated_bytes = self.allocated_bytes.saturating_sub(size_bytes);
    }

    /// Reset all working memory allocations.
    ///
    /// This is useful for clearing working memory between inference batches
    /// or when recovering from memory pressure.
    pub fn reset(&mut self) {
        self.allocated_bytes = 0;
    }
}

/// Combined memory pools for warm model loading.
///
/// Manages two isolated pools:
/// - **Model Pool**: Non-evictable, for permanent model weights
/// - **Working Pool**: Evictable, for inference activations
///
/// # Example
///
/// ```rust,ignore
/// use context_graph_embeddings::warm::memory_pool::WarmMemoryPools;
///
/// let mut pools = WarmMemoryPools::rtx_5090();
///
/// // Load a model (non-evictable)
/// pools.allocate_model("E1_Semantic", 800_000_000, 0x1000)?;
///
/// // Allocate working memory (evictable)
/// pools.allocate_working(50_000_000)?;
///
/// // Check budget compliance
/// assert!(pools.is_within_budget());
/// ```
#[derive(Debug, Clone)]
pub struct WarmMemoryPools {
    /// Non-evictable pool for model weights.
    model_pool: ModelMemoryPool,
    /// Evictable pool for working memory.
    working_pool: WorkingMemoryPool,
    /// Configuration that created these pools.
    config: WarmConfig,
}

impl WarmMemoryPools {
    /// Create new memory pools from configuration.
    ///
    /// # Arguments
    ///
    /// * `config` - Configuration specifying pool sizes
    ///
    /// # Pool Sizing
    ///
    /// - Model pool capacity: `config.vram_budget_bytes`
    /// - Working pool capacity: `config.vram_headroom_bytes`
    #[must_use]
    pub fn new(config: WarmConfig) -> Self {
        let model_pool = ModelMemoryPool::new(config.vram_budget_bytes);
        let working_pool = WorkingMemoryPool::new(config.vram_headroom_bytes);

        Self {
            model_pool,
            working_pool,
            config,
        }
    }

    /// Create pools sized for RTX 5090 (32GB VRAM).
    ///
    /// Pool allocation:
    /// - Model pool: 24GB (non-evictable)
    /// - Working pool: 8GB (evictable)
    #[must_use]
    pub fn rtx_5090() -> Self {
        Self::new(WarmConfig::default())
    }

    /// Get a reference to the configuration.
    #[must_use]
    pub fn config(&self) -> &WarmConfig {
        &self.config
    }

    /// Allocate memory for a model in the non-evictable pool.
    ///
    /// # Arguments
    ///
    /// * `model_id` - Unique identifier for the model
    /// * `size_bytes` - Size of allocation in bytes
    /// * `vram_ptr` - Raw VRAM pointer from CUDA allocation
    ///
    /// # Errors
    ///
    /// Returns `WarmError::ModelAlreadyRegistered` if model already allocated.
    /// Returns `WarmError::VramAllocationFailed` if allocation exceeds model pool capacity.
    pub fn allocate_model(
        &mut self,
        model_id: &str,
        size_bytes: usize,
        vram_ptr: u64,
    ) -> WarmResult<()> {
        self.model_pool.allocate(model_id, size_bytes, vram_ptr)
    }

    /// Free a model allocation from the non-evictable pool.
    ///
    /// # Arguments
    ///
    /// * `model_id` - Identifier of model to free
    ///
    /// # Errors
    ///
    /// Returns `WarmError::ModelNotRegistered` if model not found.
    pub fn free_model(&mut self, model_id: &str) -> WarmResult<()> {
        self.model_pool.free(model_id)?;
        Ok(())
    }

    /// Allocate working memory from the evictable pool.
    ///
    /// # Arguments
    ///
    /// * `size_bytes` - Size of allocation in bytes
    ///
    /// # Errors
    ///
    /// Returns `WarmError::WorkingMemoryExhausted` if allocation exceeds working pool capacity.
    pub fn allocate_working(&mut self, size_bytes: usize) -> WarmResult<()> {
        self.working_pool.allocate(size_bytes)
    }

    /// Free working memory from the evictable pool.
    ///
    /// # Arguments
    ///
    /// * `size_bytes` - Size to free in bytes
    ///
    /// # Note
    ///
    /// This method always succeeds. If the amount to free exceeds what
    /// is allocated, only the allocated amount will be freed.
    pub fn free_working(&mut self, size_bytes: usize) -> WarmResult<()> {
        self.working_pool.free(size_bytes);
        Ok(())
    }

    /// Get available bytes in the model pool.
    #[must_use]
    pub fn available_model_bytes(&self) -> usize {
        self.model_pool.available()
    }

    /// Get available bytes in the working pool.
    #[must_use]
    pub fn available_working_bytes(&self) -> usize {
        self.working_pool.available()
    }

    /// Get total allocated bytes across both pools.
    #[must_use]
    pub fn total_allocated_bytes(&self) -> usize {
        self.model_pool
            .allocated()
            .saturating_add(self.working_pool.allocated())
    }

    /// Check if allocations are within the configured budget.
    ///
    /// Returns `true` if:
    /// - Model pool is within `vram_budget_bytes`
    /// - Working pool is within `vram_headroom_bytes`
    #[must_use]
    pub fn is_within_budget(&self) -> bool {
        self.model_pool.allocated() <= self.config.vram_budget_bytes
            && self.working_pool.allocated() <= self.config.vram_headroom_bytes
    }

    /// Get allocation info for a specific model.
    ///
    /// # Arguments
    ///
    /// * `model_id` - Identifier of model to look up
    ///
    /// # Returns
    ///
    /// `Some(&ModelAllocation)` if found, `None` otherwise.
    #[must_use]
    pub fn get_model_allocation(&self, model_id: &str) -> Option<&ModelAllocation> {
        self.model_pool.get(model_id)
    }

    /// Get all current model allocations.
    #[must_use]
    pub fn list_model_allocations(&self) -> &[ModelAllocation] {
        self.model_pool.allocations()
    }

    /// Get the model pool capacity.
    #[must_use]
    pub fn model_pool_capacity(&self) -> usize {
        self.model_pool.capacity()
    }

    /// Get the working pool capacity.
    #[must_use]
    pub fn working_pool_capacity(&self) -> usize {
        self.working_pool.capacity()
    }

    /// Get total capacity across both pools.
    #[must_use]
    pub fn total_capacity(&self) -> usize {
        self.model_pool
            .capacity()
            .saturating_add(self.working_pool.capacity())
    }

    /// Get utilization percentage (0.0 - 1.0) for model pool.
    #[must_use]
    pub fn model_pool_utilization(&self) -> f64 {
        if self.model_pool.capacity() == 0 {
            return 0.0;
        }
        self.model_pool.allocated() as f64 / self.model_pool.capacity() as f64
    }

    /// Get utilization percentage (0.0 - 1.0) for working pool.
    #[must_use]
    pub fn working_pool_utilization(&self) -> f64 {
        if self.working_pool.capacity() == 0 {
            return 0.0;
        }
        self.working_pool.allocated() as f64 / self.working_pool.capacity() as f64
    }

    /// Reset the working memory pool.
    ///
    /// Useful for clearing between inference batches or recovering
    /// from memory pressure.
    pub fn reset_working_pool(&mut self) {
        self.working_pool.reset();
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_rtx_5090_factory_capacities() {
        let pools = WarmMemoryPools::rtx_5090();

        // 24GB for model pool
        assert_eq!(pools.model_pool_capacity(), 24 * GB);
        // 8GB for working pool
        assert_eq!(pools.working_pool_capacity(), 8 * GB);
        // 32GB total
        assert_eq!(pools.total_capacity(), 32 * GB);
    }

    #[test]
    fn test_model_allocation_and_deallocation() {
        let mut pools = WarmMemoryPools::rtx_5090();

        // Allocate a model
        let result = pools.allocate_model("E1_Semantic", 800_000_000, 0x1000);
        assert!(result.is_ok());

        // Check allocation tracking
        assert!(pools.get_model_allocation("E1_Semantic").is_some());
        let alloc = pools.get_model_allocation("E1_Semantic").unwrap();
        assert_eq!(alloc.size_bytes, 800_000_000);
        assert_eq!(alloc.vram_ptr, 0x1000);

        // Verify allocated bytes
        assert_eq!(pools.total_allocated_bytes(), 800_000_000);
        assert_eq!(pools.available_model_bytes(), 24 * GB - 800_000_000);

        // Free the model
        let result = pools.free_model("E1_Semantic");
        assert!(result.is_ok());

        // Verify freed
        assert!(pools.get_model_allocation("E1_Semantic").is_none());
        assert_eq!(pools.total_allocated_bytes(), 0);
    }

    #[test]
    fn test_budget_enforcement_model_pool() {
        let mut pools = WarmMemoryPools::rtx_5090();

        // Fill most of the model pool
        pools.allocate_model("model1", 20 * GB, 0x1000).unwrap();

        // This should succeed (still within budget)
        let result = pools.allocate_model("model2", 3 * GB, 0x2000);
        assert!(result.is_ok());

        // This should fail (exceeds capacity)
        let result = pools.allocate_model("model3", 2 * GB, 0x3000);
        assert!(result.is_err());
        match result {
            Err(WarmError::VramAllocationFailed { .. }) => (),
            _ => panic!("Expected VramAllocationFailed error"),
        }
    }

    #[test]
    fn test_working_memory_allocation() {
        let mut pools = WarmMemoryPools::rtx_5090();

        // Allocate working memory
        let result = pools.allocate_working(1 * GB);
        assert!(result.is_ok());
        assert_eq!(pools.available_working_bytes(), 7 * GB);

        // Free working memory
        pools.free_working(1 * GB).unwrap();
        assert_eq!(pools.available_working_bytes(), 8 * GB);
    }

    #[test]
    fn test_working_memory_exhaustion() {
        let mut pools = WarmMemoryPools::rtx_5090();

        // Fill the working pool
        pools.allocate_working(8 * GB).unwrap();

        // This should fail
        let result = pools.allocate_working(1);
        assert!(result.is_err());
        match result {
            Err(WarmError::WorkingMemoryExhausted { .. }) => (),
            _ => panic!("Expected WorkingMemoryExhausted error"),
        }
    }

    #[test]
    fn test_is_within_budget() {
        let mut pools = WarmMemoryPools::rtx_5090();

        // Initially within budget
        assert!(pools.is_within_budget());

        // Allocate within limits
        pools.allocate_model("model1", 20 * GB, 0x1000).unwrap();
        pools.allocate_working(6 * GB).unwrap();
        assert!(pools.is_within_budget());

        // Fill exactly to capacity - still within budget
        pools.allocate_model("model2", 4 * GB, 0x2000).unwrap();
        pools.allocate_working(2 * GB).unwrap();
        assert!(pools.is_within_budget());
    }

    #[test]
    fn test_duplicate_model_allocation() {
        let mut pools = WarmMemoryPools::rtx_5090();

        pools.allocate_model("E1_Semantic", 1 * GB, 0x1000).unwrap();

        // Duplicate should fail
        let result = pools.allocate_model("E1_Semantic", 1 * GB, 0x2000);
        assert!(result.is_err());
        match result {
            Err(WarmError::ModelAlreadyRegistered { model_id }) => {
                assert_eq!(model_id, "E1_Semantic");
            }
            _ => panic!("Expected ModelAlreadyRegistered error"),
        }
    }

    #[test]
    fn test_free_nonexistent_model() {
        let mut pools = WarmMemoryPools::rtx_5090();

        let result = pools.free_model("nonexistent");
        assert!(result.is_err());
        match result {
            Err(WarmError::ModelNotRegistered { model_id }) => {
                assert_eq!(model_id, "nonexistent");
            }
            _ => panic!("Expected ModelNotRegistered error"),
        }
    }

    #[test]
    fn test_list_model_allocations() {
        let mut pools = WarmMemoryPools::rtx_5090();

        pools.allocate_model("model1", 1 * GB, 0x1000).unwrap();
        pools.allocate_model("model2", 2 * GB, 0x2000).unwrap();
        pools.allocate_model("model3", 3 * GB, 0x3000).unwrap();

        let allocations = pools.list_model_allocations();
        assert_eq!(allocations.len(), 3);

        // Verify the allocations are tracked
        let ids: Vec<_> = allocations.iter().map(|a| a.model_id.as_str()).collect();
        assert!(ids.contains(&"model1"));
        assert!(ids.contains(&"model2"));
        assert!(ids.contains(&"model3"));
    }

    #[test]
    fn test_model_allocation_timestamp() {
        let mut pools = WarmMemoryPools::rtx_5090();
        let before = Instant::now();

        pools.allocate_model("model1", 1 * GB, 0x1000).unwrap();

        let alloc = pools.get_model_allocation("model1").unwrap();
        assert!(alloc.allocated_at >= before);
        assert!(alloc.age_secs() < 1.0); // Should be very recent
    }

    #[test]
    fn test_reset_working_pool() {
        let mut pools = WarmMemoryPools::rtx_5090();

        pools.allocate_working(5 * GB).unwrap();
        assert_eq!(pools.available_working_bytes(), 3 * GB);

        pools.reset_working_pool();
        assert_eq!(pools.available_working_bytes(), 8 * GB);
    }

    #[test]
    fn test_utilization_metrics() {
        let mut pools = WarmMemoryPools::rtx_5090();

        // Initially 0% utilization
        assert_eq!(pools.model_pool_utilization(), 0.0);
        assert_eq!(pools.working_pool_utilization(), 0.0);

        // Allocate half of each pool
        pools.allocate_model("model1", 12 * GB, 0x1000).unwrap();
        pools.allocate_working(4 * GB).unwrap();

        // Should be 50% utilization
        assert!((pools.model_pool_utilization() - 0.5).abs() < 0.001);
        assert!((pools.working_pool_utilization() - 0.5).abs() < 0.001);
    }

    #[test]
    fn test_custom_config() {
        let mut config = WarmConfig::default();
        config.vram_budget_bytes = 16 * GB;
        config.vram_headroom_bytes = 4 * GB;

        let pools = WarmMemoryPools::new(config);

        assert_eq!(pools.model_pool_capacity(), 16 * GB);
        assert_eq!(pools.working_pool_capacity(), 4 * GB);
        assert_eq!(pools.total_capacity(), 20 * GB);
    }

    #[test]
    fn test_working_memory_over_free() {
        let mut pools = WarmMemoryPools::rtx_5090();

        pools.allocate_working(1 * GB).unwrap();

        // Free more than allocated - should not underflow
        pools.free_working(10 * GB).unwrap();

        // Should be at 0, not negative
        assert_eq!(pools.working_pool.allocated(), 0);
        assert_eq!(pools.available_working_bytes(), 8 * GB);
    }

    #[test]
    fn test_zero_size_allocations() {
        let mut pools = WarmMemoryPools::rtx_5090();

        // Zero-size model allocation should succeed
        let result = pools.allocate_model("empty_model", 0, 0x1000);
        assert!(result.is_ok());

        // Zero-size working memory allocation should succeed
        let result = pools.allocate_working(0);
        assert!(result.is_ok());
    }

    #[test]
    fn test_model_pool_exhaustion_exact() {
        let mut pools = WarmMemoryPools::rtx_5090();

        // Fill exactly to capacity
        pools.allocate_model("model1", 24 * GB, 0x1000).unwrap();

        // Even 1 byte more should fail
        let result = pools.allocate_model("model2", 1, 0x2000);
        assert!(result.is_err());
    }
}
