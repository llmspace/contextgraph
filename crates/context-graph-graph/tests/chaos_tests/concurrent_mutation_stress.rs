//! Concurrent writer stress tests.
//!
//! Constitution Reference:
//! - rules: Arc<RwLock<T>> for shared state
//! - testing.types.chaos: Concurrent mutation stress testing (100 writers)
//!
//! These tests verify:
//! 1. 100 concurrent writers complete without deadlock or panic
//! 2. Mixed read/write operations are thread-safe
//! 3. Contention is handled gracefully
//! 4. Thundering herd scenarios don't cause failures

use context_graph_graph::index::gpu_memory::{GpuMemoryConfig, GpuMemoryManager, MemoryCategory};
use std::sync::{Arc, Barrier};
use std::thread;
use std::time::Duration;

/// Number of concurrent writers for stress tests.
const CONCURRENT_WRITERS: usize = 100;

/// Test 100 concurrent writers with barrier synchronization.
///
/// SYNTHETIC INPUT: 100KB budget, 100 threads each trying 512 bytes
/// EXPECTED: Some allocations succeed, no panics, no deadlocks
/// SOURCE OF TRUTH: All threads complete, stats.allocation_count consistent
#[test]
#[ignore] // Chaos test - run with --ignored
fn test_100_concurrent_writers() {
    println!("\n=== CHAOS TEST: 100 Concurrent Writers ===");

    // SYNTHETIC INPUT: 100KB budget, 100 threads each trying 512 bytes
    // Max possible: 100KB / 512 = ~200 allocations, but handles drop at thread end
    let budget = 100 * 1024;
    let allocation_size = 512;
    let manager = Arc::new(
        GpuMemoryManager::new(GpuMemoryConfig::with_budget(budget)).expect("Manager creation"),
    );
    let barrier = Arc::new(Barrier::new(CONCURRENT_WRITERS));

    println!("BEFORE: Spawning {} threads", CONCURRENT_WRITERS);
    println!(
        "BEFORE: budget={}, allocation_size={}",
        budget, allocation_size
    );

    let handles: Vec<_> = (0..CONCURRENT_WRITERS)
        .map(|i| {
            let mgr = Arc::clone(&manager);
            let bar = Arc::clone(&barrier);
            thread::spawn(move || {
                bar.wait(); // Synchronized start
                let result = mgr.allocate(allocation_size, MemoryCategory::WorkingMemory);
                // Hold allocation briefly to create contention
                if result.is_ok() {
                    thread::sleep(Duration::from_micros(100));
                }
                (i, result.is_ok())
            })
        })
        .collect();

    // Collect results
    let mut successes = 0;
    let mut failures = 0;
    for h in handles {
        let (_id, ok) = h.join().expect("Thread should not panic");
        if ok {
            successes += 1;
        } else {
            failures += 1;
        }
    }

    // SOURCE OF TRUTH: manager.stats()
    let stats = manager.stats();
    println!("AFTER: successes={}, failures={}", successes, failures);
    println!(
        "STATS: total_allocated={}, allocation_count={}",
        stats.total_allocated, stats.allocation_count
    );

    // VERIFY: No panics occurred, some succeeded
    // Note: handles drop at end of thread, so memory is freed
    assert!(
        successes > 0,
        "At least some allocations should succeed, got {} successes",
        successes
    );
    assert!(
        successes <= CONCURRENT_WRITERS,
        "Cannot exceed thread count"
    );

    // Total should be 100
    assert_eq!(
        successes + failures,
        CONCURRENT_WRITERS,
        "All threads should complete"
    );

    println!("=== PASSED: test_100_concurrent_writers ===\n");
}

/// Test mixed concurrent read/write operations.
///
/// SYNTHETIC INPUT: 1MB budget, 50 writers + 50 readers
/// EXPECTED: All threads complete without panic
/// SOURCE OF TRUTH: All threads return true (completed)
#[test]
#[ignore]
fn test_concurrent_read_write_mix() {
    println!("\n=== CHAOS TEST: Concurrent Read/Write Mix ===");

    let manager = Arc::new(
        GpuMemoryManager::new(GpuMemoryConfig::with_budget(1024 * 1024)).expect("Manager creation"),
    );
    let barrier = Arc::new(Barrier::new(100));

    println!("BEFORE: Spawning 50 writers + 50 readers");

    // 50 writers
    let writer_handles: Vec<_> = (0..50)
        .map(|i| {
            let mgr = Arc::clone(&manager);
            let bar = Arc::clone(&barrier);
            thread::spawn(move || {
                bar.wait();
                let h = mgr.allocate(1024, MemoryCategory::WorkingMemory);
                if let Ok(handle) = h {
                    thread::sleep(Duration::from_millis(1));
                    drop(handle);
                }
                (i, true) // Completed without panic
            })
        })
        .collect();

    // 50 readers
    let reader_handles: Vec<_> = (0..50)
        .map(|i| {
            let mgr = Arc::clone(&manager);
            let bar = Arc::clone(&barrier);
            thread::spawn(move || {
                bar.wait();
                // Read operations
                let _ = mgr.used();
                let _ = mgr.available();
                let _ = mgr.stats();
                let _ = mgr.is_low_memory();
                (i, true) // Completed without panic
            })
        })
        .collect();

    // Collect all - verify no panics
    let mut writer_count = 0;
    let mut reader_count = 0;

    for h in writer_handles {
        let (_id, completed) = h.join().expect("Writer thread panicked");
        if completed {
            writer_count += 1;
        }
    }
    for h in reader_handles {
        let (_id, completed) = h.join().expect("Reader thread panicked");
        if completed {
            reader_count += 1;
        }
    }

    println!(
        "AFTER: {} writers completed, {} readers completed",
        writer_count, reader_count
    );
    println!("Final stats: {:?}", manager.stats());

    assert_eq!(writer_count, 50, "All writers should complete");
    assert_eq!(reader_count, 50, "All readers should complete");

    println!("=== PASSED: test_concurrent_read_write_mix ===\n");
}

/// Test concurrent allocation contention with limited budget.
///
/// SYNTHETIC INPUT: 10KB budget, 10 threads each trying 5KB
/// EXPECTED: At most 2 succeed (10KB / 5KB = 2)
/// SOURCE OF TRUTH: successes <= 2, manager.used() <= budget
#[test]
#[ignore]
fn test_concurrent_allocation_contention() {
    println!("\n=== CHAOS TEST: Concurrent Allocation Contention ===");

    // SYNTHETIC INPUT: Limited budget, many contenders
    let budget = 10 * 1024; // 10KB
    let allocation_size = 5 * 1024; // 5KB each
                                    // Only 2 can succeed out of 10
    let num_threads = 10;

    let manager = Arc::new(
        GpuMemoryManager::new(GpuMemoryConfig::with_budget(budget)).expect("Manager creation"),
    );
    let barrier = Arc::new(Barrier::new(num_threads));

    println!(
        "BEFORE: {} threads competing for {}/{} bytes each",
        num_threads, allocation_size, budget
    );

    // Spawn threads that hold their allocations
    let handles: Vec<_> = (0..num_threads)
        .map(|i| {
            let mgr = Arc::clone(&manager);
            let bar = Arc::clone(&barrier);
            thread::spawn(move || {
                bar.wait();
                let result = mgr.allocate(allocation_size, MemoryCategory::WorkingMemory);
                if let Ok(handle) = result {
                    // Hold allocation for a moment
                    thread::sleep(Duration::from_millis(10));
                    drop(handle);
                    (i, true)
                } else {
                    (i, false)
                }
            })
        })
        .collect();

    let results: Vec<_> = handles
        .into_iter()
        .map(|h| h.join().expect("Thread panicked"))
        .collect();

    let successes: usize = results.iter().filter(|(_, ok)| *ok).count();

    // SOURCE OF TRUTH: manager.used()
    println!(
        "AFTER: {} of {} allocations succeeded",
        successes, num_threads
    );
    println!("VERIFY: used={}, budget={}", manager.used(), budget);

    // At most 2 can succeed (10KB / 5KB = 2)
    // But due to timing, might be less
    assert!(
        successes <= 2,
        "At most 2 should succeed with 5KB allocations in 10KB budget, got {}",
        successes
    );
    assert!(manager.used() <= budget, "Cannot exceed budget");

    // Memory should be freed now
    assert_eq!(manager.used(), 0, "All handles dropped, memory freed");

    println!("=== PASSED: test_concurrent_allocation_contention ===\n");
}

/// Test thundering herd scenario.
///
/// SYNTHETIC INPUT: Fill to capacity, release all, 100 threads grab immediately
/// EXPECTED: Some succeed, no panics, system remains stable
/// SOURCE OF TRUTH: used == 0 after release, some herd allocations succeed
#[test]
#[ignore]
fn test_thundering_herd_scenario() {
    println!("\n=== CHAOS TEST: Thundering Herd ===");

    let manager = Arc::new(
        GpuMemoryManager::new(GpuMemoryConfig::with_budget(1024 * 1024)).expect("Manager creation"),
    );

    // Phase 1: Fill to capacity
    println!("PHASE 1: Filling to capacity");
    let initial_handles: Vec<_> = (0..100)
        .filter_map(|_| {
            manager
                .allocate(10 * 1024, MemoryCategory::WorkingMemory)
                .ok()
        })
        .collect();

    let filled = initial_handles.len();
    println!(
        "BEFORE RELEASE: filled {} allocations, used={}",
        filled,
        manager.used()
    );

    // Phase 2: Release all at once
    drop(initial_handles);

    println!(
        "AFTER RELEASE: used={}, available={}",
        manager.used(),
        manager.available()
    );
    assert_eq!(manager.used(), 0, "All memory should be freed");

    // Phase 3: Thundering herd - 100 threads try to allocate immediately
    println!("PHASE 2: Thundering herd - 100 threads competing");
    let barrier = Arc::new(Barrier::new(100));
    let herd_handles: Vec<_> = (0..100)
        .map(|i| {
            let mgr = Arc::clone(&manager);
            let bar = Arc::clone(&barrier);
            thread::spawn(move || {
                bar.wait();
                let result = mgr.allocate(10 * 1024, MemoryCategory::WorkingMemory);
                if let Ok(handle) = result {
                    // Hold briefly
                    thread::sleep(Duration::from_millis(5));
                    drop(handle);
                    (i, true)
                } else {
                    (i, false)
                }
            })
        })
        .collect();

    let results: Vec<_> = herd_handles
        .into_iter()
        .map(|h| h.join().expect("Herd thread panicked"))
        .collect();

    let successes: usize = results.iter().filter(|(_, ok)| *ok).count();

    println!("THUNDERING HERD: {} of 100 succeeded", successes);
    assert!(successes > 0, "Some should succeed");

    // All handles dropped, verify cleanup
    assert_eq!(manager.used(), 0, "All herd allocations should be freed");

    println!("=== PASSED: test_thundering_herd_scenario ===\n");
}
