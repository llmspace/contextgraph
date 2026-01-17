//! Memory pressure scenario tests.
//!
//! Constitution Reference:
//! - perf.memory.gpu: <24GB budget tracking
//! - testing.types.chaos: Memory pressure scenario testing
//!
//! These tests verify:
//! 1. No memory leaks after thousands of allocation cycles
//! 2. System handles fragmentation patterns
//! 3. Low memory threshold warnings work correctly
//! 4. Memory accounting invariant: used + available == budget

use context_graph_graph::index::gpu_memory::{GpuMemoryConfig, GpuMemoryManager, MemoryCategory};

/// Test 10000 allocation/deallocation cycles for leak detection.
///
/// SYNTHETIC INPUT: 1MB budget, 10000 cycles of alloc/free
/// EXPECTED: used == 0 after all cycles, no leaks
/// SOURCE OF TRUTH: manager.used() == baseline after 10000 cycles
#[test]
#[ignore] // Chaos test - run with --ignored
fn test_allocation_deallocation_cycling() {
    println!("\n=== CHAOS TEST: 10000 Allocation/Deallocation Cycles ===");

    let manager =
        GpuMemoryManager::new(GpuMemoryConfig::with_budget(1024 * 1024)).expect("Manager creation");

    let baseline = manager.used();
    println!("BEFORE: baseline used={}", baseline);

    for i in 0..10000 {
        let h = manager
            .allocate(1024, MemoryCategory::WorkingMemory)
            .unwrap_or_else(|e| panic!("Allocation {} should succeed: {:?}", i, e));
        drop(h);

        // Periodic check
        if i % 1000 == 0 {
            println!("Cycle {}: used={}", i, manager.used());
        }
    }

    // SOURCE OF TRUTH: manager.used()
    println!("AFTER 10000 cycles: used={}", manager.used());
    assert_eq!(
        manager.used(),
        baseline,
        "No memory leak after cycling: expected {}, got {}",
        baseline,
        manager.used()
    );

    // Verify allocation count is 0
    let stats = manager.stats();
    assert_eq!(
        stats.allocation_count, 0,
        "No active allocations should remain"
    );

    println!("=== PASSED: test_allocation_deallocation_cycling ===\n");
}

/// Test handling of fragmentation patterns.
///
/// SYNTHETIC INPUT: 100KB budget, allocate 20 chunks, free every other one
/// EXPECTED: System tracks totals correctly (not real GPU fragmentation)
/// SOURCE OF TRUTH: available memory accounts for freed chunks
#[test]
#[ignore]
fn test_fragmentation_under_pressure() {
    println!("\n=== CHAOS TEST: Fragmentation Under Pressure ===");

    let manager =
        GpuMemoryManager::new(GpuMemoryConfig::with_budget(100 * 1024)).expect("Manager creation");

    // Allocate 20 small chunks (4KB each = 80KB total)
    let mut handles: Vec<_> = (0..20)
        .map(|i| {
            manager
                .allocate(4 * 1024, MemoryCategory::WorkingMemory)
                .unwrap_or_else(|e| panic!("Allocation {} failed: {:?}", i, e))
        })
        .collect();

    println!(
        "AFTER 20 ALLOCATIONS: used={}, available={}",
        manager.used(),
        manager.available()
    );
    assert_eq!(manager.used(), 80 * 1024, "Should have 80KB allocated");

    // Free every other one (creates fragmentation pattern) - 10 remain
    // Remove indices 18, 16, 14, 12, 10, 8, 6, 4, 2, 0 (reverse order, evens)
    for i in (0..20).rev().step_by(2) {
        handles.remove(i);
    }

    println!(
        "AFTER FRAGMENTING: used={}, available={}, handles_remaining={}",
        manager.used(),
        manager.available(),
        handles.len()
    );

    // 10 handles remain = 40KB
    assert_eq!(
        manager.used(),
        40 * 1024,
        "Should have 40KB after freeing half"
    );
    assert_eq!(handles.len(), 10, "Should have 10 handles remaining");

    // Try to allocate a larger chunk
    // Note: GpuMemoryManager tracks totals, not actual VRAM fragmentation
    // This tests the accounting system, not real GPU fragmentation
    let large_result = manager.allocate(8 * 1024, MemoryCategory::WorkingMemory);

    println!("LARGE ALLOCATION (8KB): success={}", large_result.is_ok());
    // Should succeed because we track totals (60KB available, need 8KB)
    assert!(
        large_result.is_ok(),
        "Large allocation should succeed with 60KB available"
    );

    // Clean up
    drop(handles);
    drop(large_result);
    assert_eq!(manager.used(), 0, "All memory freed");

    println!("=== PASSED: test_fragmentation_under_pressure ===\n");
}

/// Test low memory threshold warning.
///
/// SYNTHETIC INPUT: 1000 byte budget, allocate past 90% threshold
/// EXPECTED: is_low_memory() returns true when usage > 90%
/// SOURCE OF TRUTH: is_low_memory() state change
#[test]
#[ignore]
fn test_low_memory_threshold_warning() {
    println!("\n=== CHAOS TEST: Low Memory Threshold Warning ===");

    let manager = GpuMemoryManager::new(
        GpuMemoryConfig::with_budget(1000), // Use 1000 for easy percentage calc
    )
    .expect("Manager creation");

    // BEFORE: Should not be low memory
    println!(
        "BEFORE: is_low_memory={}, used={}",
        manager.is_low_memory(),
        manager.used()
    );
    assert!(
        !manager.is_low_memory(),
        "Should not be low memory initially"
    );

    // Allocate 85% - should not trigger (threshold is 90%)
    let h1 = manager
        .allocate(850, MemoryCategory::WorkingMemory)
        .expect("Allocation");
    println!(
        "AFTER 85%: is_low_memory={}, used={}",
        manager.is_low_memory(),
        manager.used()
    );
    assert!(!manager.is_low_memory(), "Should not be low memory at 85%");

    // Allocate more to pass 90% threshold (total 91%)
    let h2 = manager
        .allocate(60, MemoryCategory::WorkingMemory)
        .expect("Allocation for 91%");
    println!(
        "AFTER 91%: is_low_memory={}, used={}",
        manager.is_low_memory(),
        manager.used()
    );
    assert!(manager.is_low_memory(), "Should be low memory at 91%");

    // Drop one allocation
    drop(h2);

    // Should still be low (85%)? No, 85% < 90% threshold
    println!(
        "AFTER DROP h2: is_low_memory={}, used={}",
        manager.is_low_memory(),
        manager.used()
    );
    assert!(
        !manager.is_low_memory(),
        "Should not be low memory after dropping h2 (back to 85%)"
    );

    // Drop remaining
    drop(h1);

    // VERIFY: No longer low memory
    println!(
        "AFTER ALL DROPS: is_low_memory={}, used={}",
        manager.is_low_memory(),
        manager.used()
    );
    assert!(
        !manager.is_low_memory(),
        "Should not be low memory after all drops"
    );

    println!("=== PASSED: test_low_memory_threshold_warning ===\n");
}

/// Test memory usage accounting accuracy.
///
/// SYNTHETIC INPUT: Random allocation/deallocation sequence
/// EXPECTED: Invariant: used + available == budget always holds
/// SOURCE OF TRUTH: Verify invariant at each iteration
#[test]
#[ignore]
fn test_memory_usage_accounting_accuracy() {
    println!("\n=== CHAOS TEST: Memory Accounting Accuracy ===");

    let budget = 1024 * 1024;
    let manager =
        GpuMemoryManager::new(GpuMemoryConfig::with_budget(budget)).expect("Manager creation");

    let mut handles = Vec::new();

    // Random allocation/deallocation sequence
    for i in 0..100 {
        let size = ((i % 10) + 1) * 1024; // 1KB to 10KB

        if i % 3 == 0 && !handles.is_empty() {
            // Deallocate every 3rd iteration
            handles.pop();
        } else {
            // Allocate
            if let Ok(h) = manager.allocate(size, MemoryCategory::WorkingMemory) {
                handles.push(h);
            }
        }

        // SOURCE OF TRUTH: Verify invariant
        let used = manager.used();
        let available = manager.available();
        let actual_budget = manager.budget();

        // INVARIANT: used + available == budget
        assert_eq!(
            used + available,
            actual_budget,
            "Invariant violated at iteration {}: {} + {} != {}",
            i,
            used,
            available,
            actual_budget
        );

        if i % 20 == 0 {
            println!(
                "Iter {}: used={}, available={}, handles={}",
                i,
                used,
                available,
                handles.len()
            );
        }
    }

    // Final check with stats
    let stats = manager.stats();
    println!(
        "FINAL: allocation_count={}, total_allocated={}, handles={}",
        stats.allocation_count,
        stats.total_allocated,
        handles.len()
    );

    // Allocation count should match handle count
    assert_eq!(
        stats.allocation_count,
        handles.len(),
        "Allocation count should match handle count"
    );

    // Verify invariant one more time
    assert_eq!(
        manager.used() + manager.available(),
        manager.budget(),
        "Final invariant check failed"
    );

    // Clean up and verify
    drop(handles);
    assert_eq!(manager.used(), 0, "All memory freed");
    assert_eq!(manager.available(), budget, "Full budget available");

    println!("=== PASSED: test_memory_usage_accounting_accuracy ===\n");
}
