//! Performance benchmark tests.
//!
//! Tests latency for store and get operations to ensure performance requirements are met.

use super::common::{create_node_with_content, setup_db};
use std::time::Instant;

#[test]
fn test_store_latency() {
    let (db, _tmp) = setup_db();
    let mut latencies = Vec::with_capacity(1000);

    println!("=== STORE LATENCY TEST ===");
    println!("TRIGGER: Storing 1000 nodes and measuring latency");

    for i in 0..1000 {
        let node = create_node_with_content(&format!("Perf {}", i));
        let start = Instant::now();
        db.store_node(&node).expect("store");
        latencies.push(start.elapsed());
    }

    latencies.sort();
    let p50 = latencies[500];
    let p95 = latencies[950];
    let p99 = latencies[990];

    println!("VERIFY: p50={:?}, p95={:?}, p99={:?}", p50, p95, p99);
    assert!(
        p99 < std::time::Duration::from_millis(10),
        "p99 latency should be under 10ms, got {:?}",
        p99
    );
    println!("RESULT: PASSED");
}

#[test]
fn test_get_latency() {
    let (db, _tmp) = setup_db();
    let mut ids = Vec::with_capacity(1000);

    // First, store 1000 nodes
    for i in 0..1000 {
        let node = create_node_with_content(&format!("Perf {}", i));
        ids.push(node.id);
        db.store_node(&node).expect("store");
    }
    db.flush_all().expect("flush");

    println!("=== GET LATENCY TEST ===");
    println!("TRIGGER: Reading 1000 nodes and measuring latency");

    let mut latencies = Vec::with_capacity(1000);
    for id in &ids {
        let start = Instant::now();
        db.get_node(id).expect("get");
        latencies.push(start.elapsed());
    }

    latencies.sort();
    let p50 = latencies[500];
    let p95 = latencies[950];
    let p99 = latencies[990];

    println!("VERIFY: p50={:?}, p95={:?}, p99={:?}", p50, p95, p99);
    assert!(
        p99 < std::time::Duration::from_millis(5),
        "p99 get latency should be under 5ms, got {:?}",
        p99
    );
    println!("RESULT: PASSED");
}
