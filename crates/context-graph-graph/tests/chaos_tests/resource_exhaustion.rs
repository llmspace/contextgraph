//! Resource exhaustion and recovery tests.
//!
//! Constitution Reference:
//! - AP-001: Fail fast - graceful error handling, no panics
//! - perf.memory.gpu: <24GB budget tracking
//!
//! These tests verify:
//! 1. System remains responsive when resources exhausted
//! 2. Full recovery possible after releasing resources
//! 3. Category budgets provide isolation
//! 4. Allocation rejection messages are informative

use context_graph_graph::error::GraphError;
use context_graph_graph::index::gpu_memory::{GpuMemoryConfig, GpuMemoryManager, MemoryCategory};

/// Test graceful degradation when resources exhausted.
///
/// SYNTHETIC INPUT: 10KB budget, fill completely
/// EXPECTED: System remains responsive, new allocations fail gracefully
/// SOURCE OF TRUTH: Stats work, allocate returns Err not panic
#[test]
#[ignore] // Chaos test - run with --ignored
fn test_graceful_degradation_on_exhaustion() {
    println!("\n=== CHAOS TEST: Graceful Degradation ===");

    let manager =
        GpuMemoryManager::new(GpuMemoryConfig::with_budget(10 * 1024)).expect("Manager creation");

    // Fill to capacity
    let _h1 = manager
        .allocate(10 * 1024, MemoryCategory::WorkingMemory)
        .expect("Fill to capacity");

    println!(
        "BEFORE: Exhausted - used={}, available={}",
        manager.used(),
        manager.available()
    );

    // VERIFY: System should remain responsive
    assert_eq!(manager.available(), 0, "Should have 0 available");

    // Stats should still work
    let stats = manager.stats();
    println!(
        "STATS STILL WORK: allocation_count={}, total_allocated={}",
        stats.allocation_count, stats.total_allocated
    );
    assert_eq!(stats.allocation_count, 1, "Should have 1 allocation");
    assert_eq!(
        stats.total_allocated,
        10 * 1024,
        "Should have 10KB allocated"
    );

    // New allocations should fail gracefully (not panic)
    let result = manager.allocate(1, MemoryCategory::Other);
    println!("NEW ALLOCATION RESULT: is_err={}", result.is_err());
    assert!(result.is_err(), "Should fail gracefully");

    // Verify it's the right error type
    match result {
        Err(GraphError::GpuResourceAllocation(_)) => {
            println!("Correct error type: GpuResourceAllocation");
        }
        Err(other) => panic!("Wrong error type: {:?}", other),
        Ok(_) => panic!("Should have failed"),
    }

    // try_allocate should return None
    let opt = manager.try_allocate(1, MemoryCategory::Other);
    println!("try_allocate result: {:?}", opt.is_some());
    assert!(opt.is_none(), "try_allocate should return None");

    // Other read operations should work
    let _ = manager.used();
    let _ = manager.budget();
    let _ = manager.is_low_memory();
    println!("Read operations still work");

    println!("=== PASSED: test_graceful_degradation_on_exhaustion ===\n");
}

/// Test full recovery after resource release.
///
/// SYNTHETIC INPUT: 100KB budget, fill completely, release all
/// EXPECTED: Full capacity available after release
/// SOURCE OF TRUTH: available == budget, new full allocation succeeds
#[test]
#[ignore]
fn test_recovery_after_resource_release() {
    println!("\n=== CHAOS TEST: Recovery After Release ===");

    let budget = 100 * 1024;
    let manager =
        GpuMemoryManager::new(GpuMemoryConfig::with_budget(budget)).expect("Manager creation");

    // Fill completely with multiple allocations
    let handles: Vec<_> = (0..100)
        .filter_map(|_| manager.allocate(1024, MemoryCategory::WorkingMemory).ok())
        .collect();

    let filled = handles.len();
    println!(
        "BEFORE RELEASE: {} allocations, used={}",
        filled,
        manager.used()
    );
    assert!(filled > 0, "Should have some allocations");

    // Release all
    drop(handles);

    // SOURCE OF TRUTH: Full capacity should be available
    println!(
        "AFTER RELEASE: used={}, available={}",
        manager.used(),
        manager.available()
    );
    assert_eq!(manager.used(), 0, "All memory should be freed");
    assert_eq!(
        manager.available(),
        budget,
        "Full capacity should be available"
    );

    // Verify stats reset
    let stats = manager.stats();
    assert_eq!(stats.allocation_count, 0, "No allocations should remain");
    assert_eq!(stats.total_allocated, 0, "Nothing allocated");

    // Verify new allocations work - allocate full budget
    let new_h = manager.allocate(budget, MemoryCategory::WorkingMemory);
    assert!(
        new_h.is_ok(),
        "Should be able to allocate full budget after release"
    );
    println!("NEW ALLOCATION: Full budget allocation succeeded");

    drop(new_h);
    println!("=== PASSED: test_recovery_after_resource_release ===\n");
}

/// Test category budget exhaustion isolation.
///
/// SYNTHETIC INPUT: 100MB total, 1KB FaissIndex limit, 100MB WorkingMemory
/// EXPECTED: FaissIndex exhausted, WorkingMemory still works
/// SOURCE OF TRUTH: Different categories operate independently
#[test]
#[ignore]
fn test_category_budget_exhaustion() {
    println!("\n=== CHAOS TEST: Category Budget Exhaustion ===");

    let config = GpuMemoryConfig::with_budget(100 * 1024 * 1024) // 100MB total
        .category_budget(MemoryCategory::FaissIndex, 1024) // 1KB limit
        .category_budget(MemoryCategory::WorkingMemory, 100 * 1024 * 1024); // 100MB

    let manager = GpuMemoryManager::new(config).expect("Manager creation");

    // Exhaust FaissIndex (tiny budget)
    let _h1 = manager
        .allocate(1024, MemoryCategory::FaissIndex)
        .expect("FaissIndex allocation");

    println!("AFTER FAISS ALLOCATION: used={}", manager.used());

    // More FaissIndex should fail
    let result = manager.allocate(1, MemoryCategory::FaissIndex);
    println!("FAISS EXHAUSTED: result_is_err={}", result.is_err());
    assert!(result.is_err(), "FaissIndex should be exhausted");

    // Check error message mentions category
    if let Err(GraphError::GpuResourceAllocation(msg)) = &result {
        println!("Error message: {}", msg);
        // Error should indicate category budget issue
        let has_context = msg.contains("FaissIndex")
            || msg.contains("category")
            || msg.contains("1024")
            || msg.contains("exceed");
        assert!(has_context, "Error should indicate category issue: {}", msg);
    }

    // But WorkingMemory should work (different category)
    let h2 = manager.allocate(10 * 1024, MemoryCategory::WorkingMemory);
    assert!(h2.is_ok(), "WorkingMemory should still work");
    println!("WORKING MEMORY: allocation succeeded");

    // Verify category isolation in stats
    let stats = manager.stats();
    let faiss_usage = stats
        .category_usage
        .get(&MemoryCategory::FaissIndex)
        .copied()
        .unwrap_or(0);
    let working_usage = stats
        .category_usage
        .get(&MemoryCategory::WorkingMemory)
        .copied()
        .unwrap_or(0);

    println!(
        "CATEGORY USAGE: FaissIndex={}, WorkingMemory={}",
        faiss_usage, working_usage
    );
    assert_eq!(faiss_usage, 1024, "FaissIndex should be 1024 bytes");
    assert_eq!(working_usage, 10 * 1024, "WorkingMemory should be 10KB");

    drop(h2);
    println!("=== PASSED: test_category_budget_exhaustion ===\n");
}

/// Test allocation rejection messages include useful information.
///
/// SYNTHETIC INPUT: Various rejection scenarios
/// EXPECTED: Error messages include context (sizes, limits)
/// SOURCE OF TRUTH: Error message content
#[test]
#[ignore]
fn test_allocation_rejection_messages() {
    println!("\n=== CHAOS TEST: Allocation Rejection Messages ===");

    let manager =
        GpuMemoryManager::new(GpuMemoryConfig::with_budget(1000)).expect("Manager creation");

    // EDGE CASE 1: Request more than total budget
    println!("\nEDGE CASE 1: Request more than budget");
    println!("BEFORE: budget=1000, requesting 5000");
    let result = manager.allocate(5000, MemoryCategory::WorkingMemory);
    match result {
        Err(GraphError::GpuResourceAllocation(msg)) => {
            println!("ERROR: {}", msg);
            // Should mention the requested size or budget
            let has_context = msg.contains("5000")
                || msg.contains("1000")
                || msg.contains("exceed")
                || msg.contains("budget");
            assert!(has_context, "Message should be informative: {}", msg);
        }
        Err(e) => panic!("Wrong error type: {:?}", e),
        Ok(_) => panic!("Should have failed"),
    }

    // EDGE CASE 2: Fill then request 1 more byte
    println!("\nEDGE CASE 2: Budget exhausted, request 1 more byte");
    let _h = manager
        .allocate(1000, MemoryCategory::WorkingMemory)
        .expect("Fill budget");
    println!("BEFORE: filled budget, requesting 1 more byte");

    let result = manager.allocate(1, MemoryCategory::WorkingMemory);
    match result {
        Err(GraphError::GpuResourceAllocation(msg)) => {
            println!("ERROR: {}", msg);
            let has_context = msg.contains("1000")
                || msg.contains("exceed")
                || msg.contains("budget")
                || msg.contains("bytes");
            assert!(has_context, "Message should indicate exhaustion: {}", msg);
        }
        Err(e) => panic!("Wrong error type: {:?}", e),
        Ok(_) => panic!("Should have failed"),
    }

    // EDGE CASE 3: Zero allocation (edge case - should succeed)
    println!("\nEDGE CASE 3: Zero-size allocation");
    let manager2 =
        GpuMemoryManager::new(GpuMemoryConfig::with_budget(100)).expect("Manager creation");
    let zero_result = manager2.allocate(0, MemoryCategory::Other);
    println!("ZERO ALLOCATION: is_ok={}", zero_result.is_ok());
    assert!(zero_result.is_ok(), "Zero allocation should succeed");

    println!("\n=== PASSED: test_allocation_rejection_messages ===\n");
}
