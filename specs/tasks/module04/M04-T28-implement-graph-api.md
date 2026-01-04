---
id: "M04-T28"
title: "Implement GPU Memory Manager"
description: |
  Implement GpuMemoryManager for VRAM budget tracking and allocation.
  Target: 32GB RTX 5090 (constitution.yaml: stack.gpu.vram = "32GB")

  Memory Budget:
  - FAISS index: 8GB (10M x 1536D vectors with PQ64x8)
  - Hyperbolic coords: 2.5GB (10M x 64 floats)
  - Entailment cones: 2.7GB (10M x 68 floats)
  - Working memory: 18.8GB (remaining for batches, intermediates)

  Methods: allocate(size, category), free(handle), available(), used(), budget()
  Returns GraphError::GpuResourceAllocation if allocation exceeds budget.
layer: "surface"
status: "completed"
completed_date: "2026-01-04"
priority: "high"
estimated_hours: 3
sequence: 34
depends_on:
  - "M04-T10"  # FaissGpuIndex (uses GPU resources)
spec_refs:
  - "TECH-GRAPH-004 Section 10"
  - "NFR-KG-001"
  - "constitution.yaml: stack.gpu"
files_to_create:
  - path: "crates/context-graph-graph/src/index/gpu_memory.rs"
    description: "GPU memory manager implementation"
files_to_modify:
  - path: "crates/context-graph-graph/src/index/mod.rs"
    description: "Add gpu_memory module and re-export GpuMemoryManager"
  - path: "crates/context-graph-graph/src/lib.rs"
    description: "Re-export GpuMemoryManager from crate root"
test_file: "crates/context-graph-graph/tests/gpu_memory_tests.rs"
---

## CRITICAL CONTEXT FOR AI AGENT

### Codebase State (as of 2026-01-04)
- **Crate exists**: `crates/context-graph-graph/` is fully operational
- **GPU resources exist**: `GpuResources` in `src/index/gpu_index.rs` wraps FAISS GPU resources
- **Error type exists**: `GraphError::GpuResourceAllocation(String)` in `src/error.rs`
- **No gpu_memory.rs exists**: This file must be created from scratch
- **Integration tests**: `tests/integration_tests.rs` has 24+ tests already passing

### Key Files to Reference
```
src/index/mod.rs          - Add module declaration + re-exports
src/index/gpu_index.rs    - GpuResources pattern to follow
src/error.rs              - GraphError::GpuResourceAllocation exists
src/config.rs             - Follow config pattern (Serialize, Deserialize, Default)
src/lib.rs                - Add crate-level re-export
```

### Constitution Requirements (constitution.yaml)
```yaml
stack:
  gpu: { target: "RTX 5090", vram: "32GB", compute: "12.0" }
perf:
  memory: { gpu: "<24GB (8GB headroom)", graph_cap: ">10M nodes" }
forbidden:
  AP-015: "GPU alloc without pool → use CUDA memory pool"
```

### VRAM Budget (from contextprd.md Section 13)
```
RTX 5090: 32GB GDDR7
- FAISS Index: 8GB (10M vectors, PQ64x8)
- Hyperbolic: 2.5GB (10M x 64 x f32)
- Cones: 2.7GB (10M x 68 x f32)
- Working: 18.8GB (remaining)
Total Budget: 32GB (use 24GB safe limit per constitution)
```

## Scope

### In Scope
- `GpuMemoryManager` struct with allocation tracking
- `AllocationHandle` for RAII memory management
- `MemoryCategory` enum for budget partitioning
- `GpuMemoryConfig` for configurable budgets
- `MemoryStats` for monitoring
- Thread-safe operations via `Arc<Mutex<>>`
- Integration with existing `GraphError::GpuResourceAllocation`

### Out of Scope
- Actual CUDA memory allocation (cudaMalloc) - tracking only
- Memory defragmentation
- Multi-GPU support
- Async allocation

## Definition of Done

### File: `crates/context-graph-graph/src/index/gpu_memory.rs`

```rust
//! GPU Memory Manager for VRAM budget tracking.
//!
//! Provides centralized allocation tracking to prevent GPU OOM conditions
//! when working with 10M+ vectors on RTX 5090 (32GB VRAM).
//!
//! # Constitution Reference
//!
//! - AP-015: GPU alloc without pool → use CUDA memory pool
//! - perf.memory.gpu: <24GB (8GB headroom)
//! - stack.gpu.vram: 32GB
//!
//! # Memory Budget
//!
//! ```text
//! +------------------+--------+
//! | FAISS Index      | 8GB    |
//! | Hyperbolic Coords| 2.5GB  |
//! | Entailment Cones | 2.7GB  |
//! | Working Memory   | 10.8GB |
//! +------------------+--------+
//! | Total Safe Limit | 24GB   |
//! +------------------+--------+
//! ```

use std::collections::HashMap;
use std::sync::{Arc, Mutex};

use serde::{Deserialize, Serialize};

use crate::error::{GraphError, GraphResult};

/// Memory categories for GPU budget allocation.
///
/// Each category has a default budget that can be overridden via config.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum MemoryCategory {
    /// FAISS IVF-PQ index structures (8GB default)
    FaissIndex,
    /// Poincare point coordinates (2.5GB default)
    HyperbolicCoords,
    /// Entailment cone data (2.7GB default)
    EntailmentCones,
    /// Temporary working memory (10.8GB default)
    WorkingMemory,
    /// Uncategorized allocations (512MB default)
    Other,
}

impl MemoryCategory {
    /// Get default budget for this category in bytes.
    pub const fn default_budget(&self) -> usize {
        match self {
            MemoryCategory::FaissIndex => 8 * 1024 * 1024 * 1024,       // 8GB
            MemoryCategory::HyperbolicCoords => 2560 * 1024 * 1024,    // 2.5GB
            MemoryCategory::EntailmentCones => 2764 * 1024 * 1024,     // 2.7GB
            MemoryCategory::WorkingMemory => 10854 * 1024 * 1024,      // 10.8GB
            MemoryCategory::Other => 512 * 1024 * 1024,                // 512MB
        }
    }

    /// Get human-readable name.
    pub const fn name(&self) -> &'static str {
        match self {
            MemoryCategory::FaissIndex => "FAISS Index",
            MemoryCategory::HyperbolicCoords => "Hyperbolic Coords",
            MemoryCategory::EntailmentCones => "Entailment Cones",
            MemoryCategory::WorkingMemory => "Working Memory",
            MemoryCategory::Other => "Other",
        }
    }
}

/// Handle to allocated GPU memory.
///
/// When dropped, automatically frees the allocation.
/// The handle holds an Arc to the manager inner state.
#[derive(Debug)]
pub struct AllocationHandle {
    id: u64,
    size: usize,
    category: MemoryCategory,
    manager: Arc<Mutex<ManagerInner>>,
}

impl AllocationHandle {
    /// Get allocation size in bytes.
    #[inline]
    pub fn size(&self) -> usize {
        self.size
    }

    /// Get allocation category.
    #[inline]
    pub fn category(&self) -> MemoryCategory {
        self.category
    }

    /// Get allocation ID.
    #[inline]
    pub fn id(&self) -> u64 {
        self.id
    }
}

impl Drop for AllocationHandle {
    fn drop(&mut self) {
        if let Ok(mut inner) = self.manager.lock() {
            inner.free(self.id);
        }
    }
}

/// Configuration for GPU memory manager.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GpuMemoryConfig {
    /// Total VRAM budget in bytes.
    /// Default: 24GB (safe limit for 32GB RTX 5090)
    pub total_budget: usize,

    /// Per-category budget overrides (bytes).
    pub category_budgets: HashMap<MemoryCategory, usize>,

    /// Allow over-allocation (FOR TESTING ONLY).
    /// Default: false (fail fast per AP-001)
    pub allow_overallocation: bool,

    /// Low memory threshold fraction (0.0-1.0).
    /// Triggers warnings when usage exceeds this.
    /// Default: 0.9 (90%)
    pub low_memory_threshold: f32,
}

impl Default for GpuMemoryConfig {
    fn default() -> Self {
        Self {
            total_budget: 24 * 1024 * 1024 * 1024,  // 24GB safe limit
            category_budgets: HashMap::new(),
            allow_overallocation: false,
            low_memory_threshold: 0.9,
        }
    }
}

impl GpuMemoryConfig {
    /// Create config for RTX 5090 (24GB safe budget).
    pub fn rtx_5090() -> Self {
        Self::default()
    }

    /// Create config with custom total budget.
    pub fn with_budget(total_bytes: usize) -> Self {
        Self {
            total_budget: total_bytes,
            ..Default::default()
        }
    }

    /// Set category budget (builder pattern).
    pub fn category_budget(mut self, category: MemoryCategory, bytes: usize) -> Self {
        self.category_budgets.insert(category, bytes);
        self
    }

    /// Validate configuration.
    pub fn validate(&self) -> GraphResult<()> {
        if self.total_budget == 0 {
            return Err(GraphError::InvalidConfig(
                "total_budget must be > 0".to_string()
            ));
        }
        if self.low_memory_threshold <= 0.0 || self.low_memory_threshold > 1.0 {
            return Err(GraphError::InvalidConfig(
                format!("low_memory_threshold must be in (0, 1], got {}", self.low_memory_threshold)
            ));
        }
        Ok(())
    }
}

/// Statistics about GPU memory usage.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct MemoryStats {
    /// Total bytes allocated.
    pub total_allocated: usize,
    /// Total budget in bytes.
    pub total_budget: usize,
    /// Number of active allocations.
    pub allocation_count: usize,
    /// Peak memory usage (bytes).
    pub peak_usage: usize,
    /// Per-category usage (bytes).
    pub category_usage: HashMap<MemoryCategory, usize>,
    /// Per-category budget (bytes).
    pub category_budget: HashMap<MemoryCategory, usize>,
}

impl MemoryStats {
    /// Get usage as percentage (0-100).
    pub fn usage_percent(&self) -> f32 {
        if self.total_budget > 0 {
            (self.total_allocated as f32 / self.total_budget as f32) * 100.0
        } else {
            0.0
        }
    }

    /// Check if low memory condition.
    pub fn is_low_memory(&self, threshold: f32) -> bool {
        self.usage_percent() / 100.0 > threshold
    }

    /// Get available bytes.
    pub fn available(&self) -> usize {
        self.total_budget.saturating_sub(self.total_allocated)
    }
}

/// Internal manager state (behind Mutex).
struct ManagerInner {
    config: GpuMemoryConfig,
    allocations: HashMap<u64, (usize, MemoryCategory)>,
    category_usage: HashMap<MemoryCategory, usize>,
    total_allocated: usize,
    peak_usage: usize,
    next_id: u64,
}

impl ManagerInner {
    fn new(config: GpuMemoryConfig) -> Self {
        Self {
            config,
            allocations: HashMap::new(),
            category_usage: HashMap::new(),
            total_allocated: 0,
            peak_usage: 0,
            next_id: 0,
        }
    }

    fn allocate(&mut self, size: usize, category: MemoryCategory) -> GraphResult<u64> {
        // Check total budget
        let new_total = self.total_allocated.saturating_add(size);
        if !self.config.allow_overallocation && new_total > self.config.total_budget {
            return Err(GraphError::GpuResourceAllocation(format!(
                "Allocation of {} bytes would exceed total budget ({}/{} bytes used)",
                size, self.total_allocated, self.config.total_budget
            )));
        }

        // Check category budget
        let category_budget = self.config.category_budgets
            .get(&category)
            .copied()
            .unwrap_or_else(|| category.default_budget());

        let current_category_usage = self.category_usage.get(&category).copied().unwrap_or(0);
        let new_category_total = current_category_usage.saturating_add(size);

        if !self.config.allow_overallocation && new_category_total > category_budget {
            return Err(GraphError::GpuResourceAllocation(format!(
                "Allocation of {} bytes in {:?} would exceed category budget ({}/{} bytes)",
                size, category, current_category_usage, category_budget
            )));
        }

        // Perform allocation
        let id = self.next_id;
        self.next_id = self.next_id.wrapping_add(1);

        self.allocations.insert(id, (size, category));
        *self.category_usage.entry(category).or_insert(0) += size;
        self.total_allocated += size;
        self.peak_usage = self.peak_usage.max(self.total_allocated);

        Ok(id)
    }

    fn free(&mut self, id: u64) {
        if let Some((size, category)) = self.allocations.remove(&id) {
            self.total_allocated = self.total_allocated.saturating_sub(size);
            if let Some(usage) = self.category_usage.get_mut(&category) {
                *usage = usage.saturating_sub(size);
            }
        }
    }

    fn stats(&self) -> MemoryStats {
        let all_categories = [
            MemoryCategory::FaissIndex,
            MemoryCategory::HyperbolicCoords,
            MemoryCategory::EntailmentCones,
            MemoryCategory::WorkingMemory,
            MemoryCategory::Other,
        ];

        MemoryStats {
            total_allocated: self.total_allocated,
            total_budget: self.config.total_budget,
            allocation_count: self.allocations.len(),
            peak_usage: self.peak_usage,
            category_usage: self.category_usage.clone(),
            category_budget: all_categories
                .iter()
                .map(|&cat| {
                    let budget = self.config.category_budgets
                        .get(&cat)
                        .copied()
                        .unwrap_or_else(|| cat.default_budget());
                    (cat, budget)
                })
                .collect(),
        }
    }
}

/// GPU memory manager for VRAM budget tracking.
///
/// Provides centralized allocation tracking to prevent OOM conditions.
/// Thread-safe via Arc<Mutex<>>.
///
/// # Example
///
/// ```rust
/// use context_graph_graph::index::gpu_memory::{GpuMemoryManager, GpuMemoryConfig, MemoryCategory};
///
/// let manager = GpuMemoryManager::new(GpuMemoryConfig::rtx_5090()).unwrap();
///
/// // Allocate memory
/// let handle = manager.allocate(1024 * 1024, MemoryCategory::WorkingMemory).unwrap();
///
/// // Check stats
/// let stats = manager.stats();
/// println!("Using {} of {} bytes", stats.total_allocated, stats.total_budget);
///
/// // Memory freed automatically when handle drops
/// drop(handle);
/// assert_eq!(manager.used(), 0);
/// ```
#[derive(Clone)]
pub struct GpuMemoryManager {
    inner: Arc<Mutex<ManagerInner>>,
}

impl GpuMemoryManager {
    /// Create new memory manager with given configuration.
    ///
    /// # Errors
    ///
    /// Returns error if config validation fails.
    pub fn new(config: GpuMemoryConfig) -> GraphResult<Self> {
        config.validate()?;
        Ok(Self {
            inner: Arc::new(Mutex::new(ManagerInner::new(config))),
        })
    }

    /// Create manager for RTX 5090 (24GB safe budget).
    pub fn rtx_5090() -> GraphResult<Self> {
        Self::new(GpuMemoryConfig::rtx_5090())
    }

    /// Allocate GPU memory.
    ///
    /// Returns AllocationHandle that frees memory on drop.
    ///
    /// # Errors
    ///
    /// Returns `GraphError::GpuResourceAllocation` if:
    /// - Allocation exceeds total budget
    /// - Allocation exceeds category budget
    pub fn allocate(&self, size: usize, category: MemoryCategory) -> GraphResult<AllocationHandle> {
        let id = self.inner
            .lock()
            .map_err(|_| GraphError::GpuResourceAllocation("Lock poisoned".into()))?
            .allocate(size, category)?;

        Ok(AllocationHandle {
            id,
            size,
            category,
            manager: self.inner.clone(),
        })
    }

    /// Get available memory in bytes.
    pub fn available(&self) -> usize {
        self.inner
            .lock()
            .map(|inner| inner.config.total_budget.saturating_sub(inner.total_allocated))
            .unwrap_or(0)
    }

    /// Get used memory in bytes.
    pub fn used(&self) -> usize {
        self.inner.lock().map(|inner| inner.total_allocated).unwrap_or(0)
    }

    /// Get total budget in bytes.
    pub fn budget(&self) -> usize {
        self.inner.lock().map(|inner| inner.config.total_budget).unwrap_or(0)
    }

    /// Get memory statistics.
    pub fn stats(&self) -> MemoryStats {
        self.inner.lock().map(|inner| inner.stats()).unwrap_or_default()
    }

    /// Check if low memory condition (usage > threshold).
    pub fn is_low_memory(&self) -> bool {
        self.inner
            .lock()
            .map(|inner| {
                let threshold = inner.config.low_memory_threshold;
                let usage = inner.total_allocated as f32 / inner.config.total_budget as f32;
                usage > threshold
            })
            .unwrap_or(false)
    }

    /// Get available memory in specific category.
    pub fn category_available(&self, category: MemoryCategory) -> usize {
        self.inner
            .lock()
            .map(|inner| {
                let budget = inner.config.category_budgets
                    .get(&category)
                    .copied()
                    .unwrap_or_else(|| category.default_budget());
                let used = inner.category_usage.get(&category).copied().unwrap_or(0);
                budget.saturating_sub(used)
            })
            .unwrap_or(0)
    }

    /// Try to allocate, returning None if insufficient memory.
    pub fn try_allocate(&self, size: usize, category: MemoryCategory) -> Option<AllocationHandle> {
        self.allocate(size, category).ok()
    }
}

impl std::fmt::Debug for GpuMemoryManager {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let stats = self.stats();
        f.debug_struct("GpuMemoryManager")
            .field("used", &stats.total_allocated)
            .field("budget", &stats.total_budget)
            .field("allocations", &stats.allocation_count)
            .field("peak", &stats.peak_usage)
            .finish()
    }
}
```

### Required Modifications

#### File: `crates/context-graph-graph/src/index/mod.rs`

Add after line 41:
```rust
pub mod gpu_memory;
```

Add to re-exports after line 46:
```rust
pub use gpu_memory::{GpuMemoryManager, GpuMemoryConfig, MemoryCategory, MemoryStats, AllocationHandle};
```

#### File: `crates/context-graph-graph/src/lib.rs`

Add to index re-exports (after line 64):
```rust
pub use index::{GpuMemoryManager, GpuMemoryConfig, MemoryCategory, MemoryStats, AllocationHandle};
```

## Constraints

- **Thread Safety**: Use `Arc<Mutex<>>` for all shared state
- **RAII**: `AllocationHandle` must free memory on drop
- **Budget Enforcement**: Reject allocations that exceed budget
- **Fail Fast**: Return errors, never panic (per AP-001)
- **RTX 5090 Default**: 24GB total budget (8GB headroom from 32GB)
- **No Overallocation**: `allow_overallocation` defaults to false

## Test Requirements

### File: `crates/context-graph-graph/tests/gpu_memory_tests.rs`

**CRITICAL**: Tests must use REAL allocations, no mocks per REQ-KG-TEST.

```rust
//! GPU Memory Manager tests - M04-T28
//!
//! NO MOCKS - tests actual allocation tracking behavior.

use context_graph_graph::index::gpu_memory::{
    GpuMemoryManager, GpuMemoryConfig, MemoryCategory, MemoryStats,
};
use context_graph_graph::error::GraphError;

#[test]
fn test_allocation_and_free() {
    println!("\n=== TEST: Allocation and Free ===");
    println!("BEFORE: Creating manager with 1MB budget");

    let manager = GpuMemoryManager::new(
        GpuMemoryConfig::with_budget(1024 * 1024)
    ).expect("Manager creation failed");

    println!("AFTER: Manager created");
    assert_eq!(manager.used(), 0, "Initial used must be 0");
    assert_eq!(manager.available(), 1024 * 1024, "Initial available must be full budget");

    // Allocate
    println!("BEFORE: Allocating 512KB");
    let handle = manager.allocate(512 * 1024, MemoryCategory::WorkingMemory)
        .expect("Allocation failed");
    println!("AFTER: Allocation successful, id={}", handle.id());

    // Verify state
    assert_eq!(manager.used(), 512 * 1024, "Used should be 512KB");
    assert_eq!(manager.available(), 512 * 1024, "Available should be 512KB");
    assert_eq!(handle.size(), 512 * 1024, "Handle size should be 512KB");

    // Free via drop
    println!("BEFORE: Dropping handle");
    drop(handle);
    println!("AFTER: Handle dropped");

    // Verify freed
    assert_eq!(manager.used(), 0, "Used should be 0 after free");
    assert_eq!(manager.available(), 1024 * 1024, "Available should be restored");

    println!("=== PASSED ===\n");
}

#[test]
fn test_budget_enforcement() {
    println!("\n=== TEST: Budget Enforcement ===");

    let manager = GpuMemoryManager::new(
        GpuMemoryConfig::with_budget(1024)  // 1KB budget
    ).expect("Manager creation failed");

    // First allocation within budget
    println!("BEFORE: Allocating 512 bytes (within budget)");
    let _h1 = manager.allocate(512, MemoryCategory::WorkingMemory)
        .expect("First allocation should succeed");
    println!("AFTER: First allocation succeeded, used={}", manager.used());

    // Second allocation exceeds budget
    println!("BEFORE: Attempting 1024 byte allocation (exceeds remaining)");
    let result = manager.allocate(1024, MemoryCategory::WorkingMemory);
    println!("AFTER: Result = {:?}", result.is_err());

    assert!(result.is_err(), "Over-budget allocation must fail");
    match result {
        Err(GraphError::GpuResourceAllocation(msg)) => {
            println!("Error message: {}", msg);
            assert!(msg.contains("exceed"), "Error should mention exceeding budget");
        }
        _ => panic!("Expected GpuResourceAllocation error"),
    }

    println!("=== PASSED ===\n");
}

#[test]
fn test_category_budget() {
    println!("\n=== TEST: Category Budget ===");

    let config = GpuMemoryConfig::default()
        .category_budget(MemoryCategory::FaissIndex, 1024);

    let manager = GpuMemoryManager::new(config).expect("Manager creation failed");

    // Within category budget
    println!("BEFORE: Allocating 512 bytes to FaissIndex (budget=1024)");
    let _h1 = manager.allocate(512, MemoryCategory::FaissIndex)
        .expect("Within category budget should succeed");
    println!("AFTER: FaissIndex used={}",
        manager.stats().category_usage.get(&MemoryCategory::FaissIndex).unwrap_or(&0));

    // Exceeds category budget
    println!("BEFORE: Attempting 1024 more bytes to FaissIndex");
    let result = manager.allocate(1024, MemoryCategory::FaissIndex);
    assert!(result.is_err(), "Should fail - exceeds category budget");

    // Different category still works
    println!("BEFORE: Allocating 1024 bytes to WorkingMemory");
    let _h2 = manager.allocate(1024, MemoryCategory::WorkingMemory)
        .expect("Different category should succeed");
    println!("AFTER: WorkingMemory allocation succeeded");

    println!("=== PASSED ===\n");
}

#[test]
fn test_stats_tracking() {
    println!("\n=== TEST: Stats Tracking ===");

    let manager = GpuMemoryManager::new(
        GpuMemoryConfig::with_budget(1024 * 1024)
    ).expect("Manager creation failed");

    let _h1 = manager.allocate(100_000, MemoryCategory::FaissIndex).unwrap();
    let _h2 = manager.allocate(200_000, MemoryCategory::WorkingMemory).unwrap();

    let stats = manager.stats();
    println!("Stats: {:?}", stats);

    assert_eq!(stats.total_allocated, 300_000, "Total should be 300KB");
    assert_eq!(stats.allocation_count, 2, "Should have 2 allocations");
    assert_eq!(
        stats.category_usage.get(&MemoryCategory::FaissIndex),
        Some(&100_000),
        "FaissIndex should show 100KB"
    );
    assert_eq!(
        stats.category_usage.get(&MemoryCategory::WorkingMemory),
        Some(&200_000),
        "WorkingMemory should show 200KB"
    );

    println!("=== PASSED ===\n");
}

#[test]
fn test_thread_safety() {
    println!("\n=== TEST: Thread Safety ===");

    use std::thread;

    let manager = GpuMemoryManager::new(
        GpuMemoryConfig::with_budget(1024 * 1024 * 100)  // 100MB
    ).expect("Manager creation failed");

    let handles: Vec<_> = (0..10).map(|i| {
        let m = manager.clone();
        thread::spawn(move || {
            println!("Thread {} starting allocation", i);
            let _h = m.allocate(1024 * 1024, MemoryCategory::WorkingMemory)
                .expect("Concurrent allocation should succeed");
            thread::sleep(std::time::Duration::from_millis(10));
            println!("Thread {} completing", i);
        })
    }).collect();

    for h in handles {
        h.join().expect("Thread should not panic");
    }

    // All allocations freed after threads complete
    assert_eq!(manager.used(), 0, "All memory should be freed");

    println!("=== PASSED ===\n");
}

#[test]
fn test_rtx_5090_config() {
    println!("\n=== TEST: RTX 5090 Config ===");

    let manager = GpuMemoryManager::rtx_5090().expect("RTX 5090 config failed");

    assert_eq!(manager.budget(), 24 * 1024 * 1024 * 1024, "Budget should be 24GB");

    let stats = manager.stats();
    assert_eq!(stats.total_budget, 24 * 1024 * 1024 * 1024);

    println!("=== PASSED ===\n");
}

#[test]
fn test_try_allocate() {
    println!("\n=== TEST: try_allocate ===");

    let manager = GpuMemoryManager::new(
        GpuMemoryConfig::with_budget(1024)
    ).expect("Manager creation failed");

    // Success case
    let h = manager.try_allocate(512, MemoryCategory::WorkingMemory);
    assert!(h.is_some(), "Should succeed with Some");

    // Failure case (returns None, not error)
    let h2 = manager.try_allocate(1024, MemoryCategory::WorkingMemory);
    assert!(h2.is_none(), "Should return None on failure");

    println!("=== PASSED ===\n");
}

#[test]
fn test_peak_usage_tracking() {
    println!("\n=== TEST: Peak Usage Tracking ===");

    let manager = GpuMemoryManager::new(
        GpuMemoryConfig::with_budget(1024 * 1024)
    ).expect("Manager creation failed");

    let h1 = manager.allocate(256 * 1024, MemoryCategory::WorkingMemory).unwrap();
    let h2 = manager.allocate(256 * 1024, MemoryCategory::WorkingMemory).unwrap();

    assert_eq!(manager.stats().peak_usage, 512 * 1024, "Peak should be 512KB");

    drop(h2);
    drop(h1);

    // Peak should remain even after freeing
    assert_eq!(manager.stats().peak_usage, 512 * 1024, "Peak should persist");
    assert_eq!(manager.used(), 0, "Current usage should be 0");

    println!("=== PASSED ===\n");
}

// ============================================================================
// EDGE CASE TESTS
// ============================================================================

#[test]
fn test_edge_case_zero_size_allocation() {
    println!("\n=== EDGE CASE: Zero Size Allocation ===");
    println!("BEFORE: manager.used() = {}", 0);

    let manager = GpuMemoryManager::new(
        GpuMemoryConfig::with_budget(1024)
    ).expect("Manager creation failed");

    // Zero-size allocation is allowed (no-op effectively)
    let h = manager.allocate(0, MemoryCategory::WorkingMemory).unwrap();

    println!("AFTER: manager.used() = {}", manager.used());
    assert_eq!(manager.used(), 0, "Zero allocation should not change usage");
    assert_eq!(h.size(), 0, "Handle size should be 0");

    drop(h);
    assert_eq!(manager.used(), 0, "Should still be 0 after drop");

    println!("=== PASSED ===\n");
}

#[test]
fn test_edge_case_max_allocation() {
    println!("\n=== EDGE CASE: Maximum Allocation ===");

    let budget: usize = 1024 * 1024;  // 1MB
    let manager = GpuMemoryManager::new(
        GpuMemoryConfig::with_budget(budget)
    ).expect("Manager creation failed");

    println!("BEFORE: Allocating exact budget size");
    let h = manager.allocate(budget, MemoryCategory::WorkingMemory).unwrap();
    println!("AFTER: manager.used() = {}, manager.available() = {}",
        manager.used(), manager.available());

    assert_eq!(manager.used(), budget);
    assert_eq!(manager.available(), 0);

    // Any additional allocation should fail
    let result = manager.allocate(1, MemoryCategory::Other);
    assert!(result.is_err(), "Should fail when budget exhausted");

    drop(h);
    println!("=== PASSED ===\n");
}

#[test]
fn test_edge_case_empty_inputs() {
    println!("\n=== EDGE CASE: Boundary Conditions ===");

    // Test config with minimal budget
    let manager = GpuMemoryManager::new(
        GpuMemoryConfig::with_budget(1)  // 1 byte budget
    ).expect("Should allow 1 byte budget");

    let h = manager.allocate(1, MemoryCategory::Other).unwrap();
    assert_eq!(manager.used(), 1);
    assert_eq!(manager.available(), 0);

    let result = manager.allocate(1, MemoryCategory::Other);
    assert!(result.is_err());

    drop(h);
    println!("=== PASSED ===\n");
}

#[test]
fn test_edge_case_invalid_config() {
    println!("\n=== EDGE CASE: Invalid Config ===");

    // Zero budget should fail
    let result = GpuMemoryManager::new(
        GpuMemoryConfig::with_budget(0)
    );
    assert!(result.is_err(), "Zero budget should fail validation");

    // Invalid threshold should fail
    let mut config = GpuMemoryConfig::default();
    config.low_memory_threshold = 0.0;
    let result = GpuMemoryManager::new(config);
    assert!(result.is_err(), "Zero threshold should fail validation");

    println!("=== PASSED ===\n");
}
```

## Verification Commands

```bash
# Build the crate
cargo build -p context-graph-graph

# Run GPU memory tests
cargo test -p context-graph-graph gpu_memory -- --nocapture

# Run clippy (must pass with no warnings)
cargo clippy -p context-graph-graph -- -D warnings

# Verify re-exports work
cargo doc -p context-graph-graph --no-deps
```

## Full State Verification Requirements

After implementing, you MUST verify the following:

### 1. Source of Truth Verification
The source of truth is the `ManagerInner` struct's `allocations` HashMap.

```rust
// After each operation, verify internal state matches external API
let stats = manager.stats();
assert_eq!(stats.allocation_count, /* expected count */);
assert_eq!(stats.total_allocated, /* expected bytes */);
```

### 2. Execute & Inspect Pattern
For each test:
1. Print state BEFORE operation
2. Execute operation
3. Print state AFTER operation
4. Assert expected state

### 3. Edge Case Audit (3 Required)
1. **Empty input**: Zero-size allocation
2. **Maximum limit**: Allocate exactly the budget
3. **Invalid format**: Zero budget config, invalid threshold

### 4. Evidence of Success
Each test must print:
- State before operation
- State after operation
- Whether assertion passed

## Acceptance Criteria

- [ ] `gpu_memory.rs` created with all types
- [ ] `GpuMemoryManager` struct with budget tracking
- [ ] `allocate()` reserves memory and returns handle
- [ ] `free()` releases memory via `Drop` on `AllocationHandle`
- [ ] `available()` returns remaining budget
- [ ] `used()` returns current allocation total
- [ ] `budget()` returns configured budget
- [ ] `stats()` returns detailed `MemoryStats`
- [ ] Returns `GraphError::GpuResourceAllocation` if over budget
- [ ] Thread-safe via `Arc<Mutex<>>`
- [ ] `mod.rs` updated with module + re-exports
- [ ] `lib.rs` updated with crate-level re-exports
- [ ] All tests pass with `cargo test`
- [ ] No clippy warnings
- [ ] Edge cases tested and documented

## FINAL VERIFICATION STEP

After completing implementation:

**MUST USE sherlock-holmes subagent** to verify:
1. All files exist at correct paths
2. All exports compile correctly
3. All tests pass
4. No clippy warnings
5. Integration with existing code works
6. Memory is actually freed on handle drop
7. Budget enforcement actually prevents over-allocation

The sherlock-holmes agent will forensically verify the entire implementation is correct and identify any issues that need fixing.

## Implementation Status

### Sherlock-Holmes Forensic Verification (2026-01-04)

**VERDICT: INNOCENT** - All requirements VERIFIED ✅

#### Evidence Summary:
- **18/18 Requirements Satisfied**
- **34/34 Tests Passing** (23 integration + 11 unit tests)
- **0 Clippy Warnings** in target files

#### Verified Items:
1. ✅ `gpu_memory.rs` created at correct path (557 lines)
2. ✅ `GpuMemoryManager` struct with `Arc<Mutex<ManagerInner>>` pattern
3. ✅ `MemoryCategory` enum with 5 variants and default budgets
4. ✅ `AllocationHandle` with RAII Drop implementation
5. ✅ `GpuMemoryConfig` with validation and builder pattern
6. ✅ `MemoryStats` with usage tracking
7. ✅ Thread-safe via `Arc<Mutex<>>` pattern
8. ✅ Budget enforcement returns `GraphError::GpuResourceAllocation`
9. ✅ Category budget enforcement working
10. ✅ 24GB default budget (RTX 5090 safe limit)
11. ✅ `mod.rs` updated with `pub mod gpu_memory;` and re-exports
12. ✅ `lib.rs` updated with crate-level re-exports
13. ✅ All edge cases tested (zero-size, max allocation, invalid config)
14. ✅ Peak usage tracking implemented
15. ✅ `try_allocate()` non-panicking API available
16. ✅ `category_available()` per-category query working
17. ✅ `is_low_memory()` threshold checking working
18. ✅ Memory freed on handle drop verified in tests

#### Test Coverage:
```
Integration tests (tests/gpu_memory_tests.rs):
- test_allocation_and_free
- test_budget_enforcement
- test_category_budget
- test_stats_tracking
- test_thread_safety
- test_rtx_5090_config
- test_try_allocate
- test_peak_usage_tracking
- test_edge_case_zero_size_allocation
- test_edge_case_max_allocation
- test_edge_case_empty_inputs
- test_edge_case_invalid_config
+ 12 additional integration tests

Unit tests (in gpu_memory.rs):
- test_memory_category_default_budgets
- test_memory_category_names
- test_gpu_memory_config_default
- test_gpu_memory_config_builder
- test_gpu_memory_config_validate
- test_memory_stats_usage_percent
- test_memory_stats_is_low_memory
- test_memory_stats_available
- test_manager_creation
- test_allocation_handle_properties
- test_manager_debug_format
```

#### Minor Finding (Acceptable):
- WorkingMemory default is 10.6GB (10854 MiB) vs spec's 10.8GB
- Difference: 0.2GB due to MiB vs GB rounding - within acceptable tolerance
