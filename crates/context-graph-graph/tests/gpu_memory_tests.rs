//! GPU Memory Manager tests - M04-T28
//!
//! NO MOCKS - tests actual allocation tracking behavior.
//!
//! # Full State Verification Requirements
//!
//! After each operation, verify internal state matches external API:
//! - Source of truth: `ManagerInner.allocations` HashMap
//! - Each test prints state BEFORE and AFTER operations
//! - Edge cases: zero-size, max allocation, invalid config
//!
//! # Constitution Reference
//!
//! - AP-001: Fail fast, never unwrap() in prod
//! - AP-015: GPU alloc without pool → use CUDA memory pool
//! - perf.memory.gpu: <24GB (8GB headroom)

use context_graph_graph::error::GraphError;
use context_graph_graph::index::gpu_memory::{
    GpuMemoryConfig, GpuMemoryManager, MemoryCategory, MemoryStats,
};

// ============================================================================
// BASIC FUNCTIONALITY TESTS
// ============================================================================

#[test]
fn test_allocation_and_free() {
    println!("\n=== TEST: Allocation and Free ===");
    println!("BEFORE: Creating manager with 1MB budget");

    let manager = GpuMemoryManager::new(GpuMemoryConfig::with_budget(1024 * 1024))
        .expect("Manager creation failed");

    println!("AFTER: Manager created");
    assert_eq!(manager.used(), 0, "Initial used must be 0");
    assert_eq!(
        manager.available(),
        1024 * 1024,
        "Initial available must be full budget"
    );

    // Allocate
    println!("BEFORE: Allocating 512KB");
    let handle = manager
        .allocate(512 * 1024, MemoryCategory::WorkingMemory)
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
    assert_eq!(
        manager.available(),
        1024 * 1024,
        "Available should be restored"
    );

    println!("=== PASSED ===\n");
}

#[test]
fn test_budget_enforcement() {
    println!("\n=== TEST: Budget Enforcement ===");

    let manager =
        GpuMemoryManager::new(GpuMemoryConfig::with_budget(1024)).expect("Manager creation failed");

    // First allocation within budget
    println!("BEFORE: Allocating 512 bytes (within budget)");
    let _h1 = manager
        .allocate(512, MemoryCategory::WorkingMemory)
        .expect("First allocation should succeed");
    println!(
        "AFTER: First allocation succeeded, used={}",
        manager.used()
    );

    // Second allocation exceeds budget
    println!("BEFORE: Attempting 1024 byte allocation (exceeds remaining)");
    let result = manager.allocate(1024, MemoryCategory::WorkingMemory);
    println!("AFTER: Result = {:?}", result.is_err());

    assert!(result.is_err(), "Over-budget allocation must fail");
    match result {
        Err(GraphError::GpuResourceAllocation(msg)) => {
            println!("Error message: {}", msg);
            assert!(
                msg.contains("exceed"),
                "Error should mention exceeding budget"
            );
        }
        _ => panic!("Expected GpuResourceAllocation error"),
    }

    println!("=== PASSED ===\n");
}

#[test]
fn test_category_budget() {
    println!("\n=== TEST: Category Budget ===");

    let config =
        GpuMemoryConfig::default().category_budget(MemoryCategory::FaissIndex, 1024);

    let manager = GpuMemoryManager::new(config).expect("Manager creation failed");

    // Within category budget
    println!("BEFORE: Allocating 512 bytes to FaissIndex (budget=1024)");
    let _h1 = manager
        .allocate(512, MemoryCategory::FaissIndex)
        .expect("Within category budget should succeed");
    println!(
        "AFTER: FaissIndex used={}",
        manager
            .stats()
            .category_usage
            .get(&MemoryCategory::FaissIndex)
            .unwrap_or(&0)
    );

    // Exceeds category budget
    println!("BEFORE: Attempting 1024 more bytes to FaissIndex");
    let result = manager.allocate(1024, MemoryCategory::FaissIndex);
    assert!(result.is_err(), "Should fail - exceeds category budget");

    // Different category still works
    println!("BEFORE: Allocating 1024 bytes to WorkingMemory");
    let _h2 = manager
        .allocate(1024, MemoryCategory::WorkingMemory)
        .expect("Different category should succeed");
    println!("AFTER: WorkingMemory allocation succeeded");

    println!("=== PASSED ===\n");
}

#[test]
fn test_stats_tracking() {
    println!("\n=== TEST: Stats Tracking ===");

    let manager = GpuMemoryManager::new(GpuMemoryConfig::with_budget(1024 * 1024))
        .expect("Manager creation failed");

    let _h1 = manager
        .allocate(100_000, MemoryCategory::FaissIndex)
        .expect("Allocation failed");
    let _h2 = manager
        .allocate(200_000, MemoryCategory::WorkingMemory)
        .expect("Allocation failed");

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

    let manager = GpuMemoryManager::new(GpuMemoryConfig::with_budget(1024 * 1024 * 100))
        .expect("Manager creation failed");

    let handles: Vec<_> = (0..10)
        .map(|i| {
            let m = manager.clone();
            thread::spawn(move || {
                println!("Thread {} starting allocation", i);
                let _h = m
                    .allocate(1024 * 1024, MemoryCategory::WorkingMemory)
                    .expect("Concurrent allocation should succeed");
                thread::sleep(std::time::Duration::from_millis(10));
                println!("Thread {} completing", i);
            })
        })
        .collect();

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

    assert_eq!(
        manager.budget(),
        24 * 1024 * 1024 * 1024,
        "Budget should be 24GB"
    );

    let stats = manager.stats();
    assert_eq!(stats.total_budget, 24 * 1024 * 1024 * 1024);

    println!("=== PASSED ===\n");
}

#[test]
fn test_try_allocate() {
    println!("\n=== TEST: try_allocate ===");

    let manager =
        GpuMemoryManager::new(GpuMemoryConfig::with_budget(1024)).expect("Manager creation failed");

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

    let manager = GpuMemoryManager::new(GpuMemoryConfig::with_budget(1024 * 1024))
        .expect("Manager creation failed");

    let h1 = manager
        .allocate(256 * 1024, MemoryCategory::WorkingMemory)
        .expect("Allocation failed");
    let h2 = manager
        .allocate(256 * 1024, MemoryCategory::WorkingMemory)
        .expect("Allocation failed");

    assert_eq!(
        manager.stats().peak_usage,
        512 * 1024,
        "Peak should be 512KB"
    );

    drop(h2);
    drop(h1);

    // Peak should remain even after freeing
    assert_eq!(
        manager.stats().peak_usage,
        512 * 1024,
        "Peak should persist"
    );
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

    let manager =
        GpuMemoryManager::new(GpuMemoryConfig::with_budget(1024)).expect("Manager creation failed");

    // Zero-size allocation is allowed (no-op effectively)
    let h = manager
        .allocate(0, MemoryCategory::WorkingMemory)
        .expect("Zero allocation should succeed");

    println!("AFTER: manager.used() = {}", manager.used());
    assert_eq!(
        manager.used(),
        0,
        "Zero allocation should not change usage"
    );
    assert_eq!(h.size(), 0, "Handle size should be 0");

    drop(h);
    assert_eq!(manager.used(), 0, "Should still be 0 after drop");

    println!("=== PASSED ===\n");
}

#[test]
fn test_edge_case_max_allocation() {
    println!("\n=== EDGE CASE: Maximum Allocation ===");

    let budget: usize = 1024 * 1024; // 1MB
    let manager =
        GpuMemoryManager::new(GpuMemoryConfig::with_budget(budget)).expect("Manager creation failed");

    println!("BEFORE: Allocating exact budget size");
    let h = manager
        .allocate(budget, MemoryCategory::WorkingMemory)
        .expect("Exact budget allocation should succeed");
    println!(
        "AFTER: manager.used() = {}, manager.available() = {}",
        manager.used(),
        manager.available()
    );

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
    let manager = GpuMemoryManager::new(GpuMemoryConfig::with_budget(1))
        .expect("Should allow 1 byte budget");

    let h = manager
        .allocate(1, MemoryCategory::Other)
        .expect("Single byte allocation should succeed");
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
    let result = GpuMemoryManager::new(GpuMemoryConfig::with_budget(0));
    assert!(result.is_err(), "Zero budget should fail validation");

    // Invalid threshold should fail
    let mut config = GpuMemoryConfig::default();
    config.low_memory_threshold = 0.0;
    let result = GpuMemoryManager::new(config);
    assert!(result.is_err(), "Zero threshold should fail validation");

    println!("=== PASSED ===\n");
}

// ============================================================================
// LOW MEMORY DETECTION TESTS
// ============================================================================

#[test]
fn test_low_memory_detection() {
    println!("\n=== TEST: Low Memory Detection ===");

    let mut config = GpuMemoryConfig::with_budget(1000);
    config.low_memory_threshold = 0.8; // 80%

    let manager = GpuMemoryManager::new(config).expect("Manager creation failed");

    // Below threshold
    let _h1 = manager
        .allocate(700, MemoryCategory::WorkingMemory)
        .expect("Allocation failed");
    assert!(
        !manager.is_low_memory(),
        "70% usage should not be low memory"
    );

    // Above threshold
    let _h2 = manager
        .allocate(200, MemoryCategory::WorkingMemory)
        .expect("Allocation failed");
    assert!(manager.is_low_memory(), "90% usage should be low memory");

    println!("=== PASSED ===\n");
}

// ============================================================================
// CATEGORY AVAILABLE TESTS
// ============================================================================

#[test]
fn test_category_available() {
    println!("\n=== TEST: Category Available ===");

    let config =
        GpuMemoryConfig::default().category_budget(MemoryCategory::FaissIndex, 1000);

    let manager = GpuMemoryManager::new(config).expect("Manager creation failed");

    // Initial available
    assert_eq!(
        manager.category_available(MemoryCategory::FaissIndex),
        1000
    );

    // After allocation
    let _h = manager
        .allocate(400, MemoryCategory::FaissIndex)
        .expect("Allocation failed");
    assert_eq!(
        manager.category_available(MemoryCategory::FaissIndex),
        600
    );

    println!("=== PASSED ===\n");
}

// ============================================================================
// MULTIPLE ALLOCATION TESTS
// ============================================================================

#[test]
fn test_multiple_allocations_different_categories() {
    println!("\n=== TEST: Multiple Allocations Different Categories ===");

    let config = GpuMemoryConfig::with_budget(10_000_000)
        .category_budget(MemoryCategory::FaissIndex, 1_000_000)
        .category_budget(MemoryCategory::HyperbolicCoords, 2_000_000)
        .category_budget(MemoryCategory::EntailmentCones, 3_000_000);

    let manager = GpuMemoryManager::new(config).expect("Manager creation failed");

    println!("BEFORE: Allocating across multiple categories");

    let h1 = manager
        .allocate(500_000, MemoryCategory::FaissIndex)
        .expect("FaissIndex allocation failed");
    let h2 = manager
        .allocate(1_000_000, MemoryCategory::HyperbolicCoords)
        .expect("HyperbolicCoords allocation failed");
    let h3 = manager
        .allocate(1_500_000, MemoryCategory::EntailmentCones)
        .expect("EntailmentCones allocation failed");

    let stats = manager.stats();
    println!("AFTER: total_allocated={}", stats.total_allocated);
    println!("  FaissIndex: {:?}", stats.category_usage.get(&MemoryCategory::FaissIndex));
    println!("  HyperbolicCoords: {:?}", stats.category_usage.get(&MemoryCategory::HyperbolicCoords));
    println!("  EntailmentCones: {:?}", stats.category_usage.get(&MemoryCategory::EntailmentCones));

    assert_eq!(stats.total_allocated, 3_000_000);
    assert_eq!(stats.allocation_count, 3);

    // Verify per-category
    assert_eq!(
        stats.category_usage.get(&MemoryCategory::FaissIndex),
        Some(&500_000)
    );
    assert_eq!(
        stats.category_usage.get(&MemoryCategory::HyperbolicCoords),
        Some(&1_000_000)
    );
    assert_eq!(
        stats.category_usage.get(&MemoryCategory::EntailmentCones),
        Some(&1_500_000)
    );

    // Drop one and verify
    drop(h2);
    let stats = manager.stats();
    assert_eq!(stats.total_allocated, 2_000_000);
    assert_eq!(stats.allocation_count, 2);
    assert_eq!(
        stats.category_usage.get(&MemoryCategory::HyperbolicCoords),
        Some(&0)
    );

    drop(h1);
    drop(h3);

    assert_eq!(manager.used(), 0);
    assert_eq!(manager.stats().allocation_count, 0);

    println!("=== PASSED ===\n");
}

// ============================================================================
// ALLOCATION HANDLE TESTS
// ============================================================================

#[test]
fn test_allocation_handle_properties() {
    println!("\n=== TEST: Allocation Handle Properties ===");

    let manager = GpuMemoryManager::new(GpuMemoryConfig::with_budget(1_000_000))
        .expect("Manager creation failed");

    let h1 = manager
        .allocate(1000, MemoryCategory::FaissIndex)
        .expect("Allocation failed");
    let h2 = manager
        .allocate(2000, MemoryCategory::WorkingMemory)
        .expect("Allocation failed");

    // Verify handle properties
    assert_eq!(h1.size(), 1000);
    assert_eq!(h1.category(), MemoryCategory::FaissIndex);

    assert_eq!(h2.size(), 2000);
    assert_eq!(h2.category(), MemoryCategory::WorkingMemory);

    // IDs should be different
    assert_ne!(h1.id(), h2.id());

    println!("Handle 1: id={}, size={}, category={:?}", h1.id(), h1.size(), h1.category());
    println!("Handle 2: id={}, size={}, category={:?}", h2.id(), h2.size(), h2.category());

    println!("=== PASSED ===\n");
}

// ============================================================================
// DEBUG FORMAT TESTS
// ============================================================================

#[test]
fn test_debug_format() {
    println!("\n=== TEST: Debug Format ===");

    let manager = GpuMemoryManager::new(GpuMemoryConfig::with_budget(1_000_000))
        .expect("Manager creation failed");

    let _h = manager
        .allocate(500_000, MemoryCategory::WorkingMemory)
        .expect("Allocation failed");

    let debug_str = format!("{:?}", manager);
    println!("Debug output: {}", debug_str);

    assert!(debug_str.contains("GpuMemoryManager"));
    assert!(debug_str.contains("used"));
    assert!(debug_str.contains("budget"));

    println!("=== PASSED ===\n");
}

// ============================================================================
// MEMORY STATS TESTS
// ============================================================================

#[test]
fn test_memory_stats_usage_percent() {
    println!("\n=== TEST: Memory Stats Usage Percent ===");

    let manager = GpuMemoryManager::new(GpuMemoryConfig::with_budget(1000))
        .expect("Manager creation failed");

    let _h = manager
        .allocate(500, MemoryCategory::WorkingMemory)
        .expect("Allocation failed");

    let stats = manager.stats();
    let percent = stats.usage_percent();

    println!("Usage percent: {}%", percent);
    assert!((percent - 50.0).abs() < 0.1, "Should be approximately 50%");

    println!("=== PASSED ===\n");
}

#[test]
fn test_memory_stats_category_budget() {
    println!("\n=== TEST: Memory Stats Category Budget ===");

    let manager = GpuMemoryManager::new(GpuMemoryConfig::default())
        .expect("Manager creation failed");

    let stats = manager.stats();

    // Verify all categories have budgets
    assert!(stats.category_budget.contains_key(&MemoryCategory::FaissIndex));
    assert!(stats.category_budget.contains_key(&MemoryCategory::HyperbolicCoords));
    assert!(stats.category_budget.contains_key(&MemoryCategory::EntailmentCones));
    assert!(stats.category_budget.contains_key(&MemoryCategory::WorkingMemory));
    assert!(stats.category_budget.contains_key(&MemoryCategory::Other));

    // Verify default budgets
    assert_eq!(
        stats.category_budget.get(&MemoryCategory::FaissIndex),
        Some(&(8 * 1024 * 1024 * 1024))
    );

    println!("Category budgets verified");
    println!("=== PASSED ===\n");
}

// ============================================================================
// CLONE TESTS
// ============================================================================

#[test]
fn test_manager_clone() {
    println!("\n=== TEST: Manager Clone ===");

    let manager = GpuMemoryManager::new(GpuMemoryConfig::with_budget(1_000_000))
        .expect("Manager creation failed");

    let _h = manager
        .allocate(500_000, MemoryCategory::WorkingMemory)
        .expect("Allocation failed");

    // Clone manager
    let manager2 = manager.clone();

    // Both should see the same state
    assert_eq!(manager.used(), manager2.used());
    assert_eq!(manager.budget(), manager2.budget());

    // Allocate on clone, original should see it
    let _h2 = manager2
        .allocate(100_000, MemoryCategory::FaissIndex)
        .expect("Allocation on clone failed");

    assert_eq!(manager.used(), 600_000);
    assert_eq!(manager2.used(), 600_000);

    println!("=== PASSED ===\n");
}

// ============================================================================
// SERIALIZATION TESTS
// ============================================================================

#[test]
fn test_config_serialization() {
    println!("\n=== TEST: Config Serialization ===");

    let config = GpuMemoryConfig::with_budget(1_000_000)
        .category_budget(MemoryCategory::FaissIndex, 500_000);

    // Serialize to JSON
    let json = serde_json::to_string_pretty(&config).expect("Serialization failed");
    println!("Config JSON:\n{}", json);

    // Deserialize back
    let config2: GpuMemoryConfig =
        serde_json::from_str(&json).expect("Deserialization failed");

    assert_eq!(config.total_budget, config2.total_budget);
    assert_eq!(
        config.category_budgets.get(&MemoryCategory::FaissIndex),
        config2.category_budgets.get(&MemoryCategory::FaissIndex)
    );

    println!("=== PASSED ===\n");
}

#[test]
fn test_stats_serialization() {
    println!("\n=== TEST: Stats Serialization ===");

    let manager = GpuMemoryManager::new(GpuMemoryConfig::with_budget(1_000_000))
        .expect("Manager creation failed");

    let _h = manager
        .allocate(500_000, MemoryCategory::WorkingMemory)
        .expect("Allocation failed");

    let stats = manager.stats();

    // Serialize to JSON
    let json = serde_json::to_string_pretty(&stats).expect("Serialization failed");
    println!("Stats JSON:\n{}", json);

    // Deserialize back
    let stats2: MemoryStats = serde_json::from_str(&json).expect("Deserialization failed");

    assert_eq!(stats.total_allocated, stats2.total_allocated);
    assert_eq!(stats.total_budget, stats2.total_budget);

    println!("=== PASSED ===\n");
}

// ============================================================================
// CATEGORY ENUM TESTS
// ============================================================================

#[test]
fn test_memory_category_serialization() {
    println!("\n=== TEST: Memory Category Serialization ===");

    let categories = [
        MemoryCategory::FaissIndex,
        MemoryCategory::HyperbolicCoords,
        MemoryCategory::EntailmentCones,
        MemoryCategory::WorkingMemory,
        MemoryCategory::Other,
    ];

    for cat in categories.iter() {
        let json = serde_json::to_string(cat).expect("Serialization failed");
        println!("{:?} -> {}", cat, json);

        let deserialized: MemoryCategory =
            serde_json::from_str(&json).expect("Deserialization failed");
        assert_eq!(*cat, deserialized);
    }

    println!("=== PASSED ===\n");
}
