//! Concurrent access tests.
//!
//! Tests thread-safe read and write operations.

use super::common::{create_node_with_content, create_test_node, setup_db};
use std::sync::Arc;
use std::thread;

#[test]
fn test_concurrent_reads() {
    let (db, _tmp) = setup_db();
    let db = Arc::new(db);
    let node = create_test_node();
    let node_id = node.id;
    db.store_node(&node).expect("store");

    println!("=== CONCURRENT READS TEST ===");
    println!("TRIGGER: Spawning 100 concurrent read threads");

    let handles: Vec<_> = (0..100)
        .map(|_| {
            let db = Arc::clone(&db);
            thread::spawn(move || db.get_node(&node_id).is_ok())
        })
        .collect();

    let success_count: usize = handles
        .into_iter()
        .map(|h| if h.join().unwrap() { 1 } else { 0 })
        .sum();

    println!("VERIFY: {}/100 reads succeeded", success_count);
    assert_eq!(success_count, 100, "all 100 reads should succeed");
    println!("RESULT: PASSED");
}

#[test]
fn test_concurrent_writes() {
    let (db, _tmp) = setup_db();
    let db = Arc::new(db);

    println!("=== CONCURRENT WRITES TEST ===");
    println!("TRIGGER: Spawning 50 concurrent write threads");

    let handles: Vec<_> = (0..50)
        .map(|i| {
            let db = Arc::clone(&db);
            thread::spawn(move || {
                let node = create_node_with_content(&format!("Node {}", i));
                let id = node.id;
                (id, db.store_node(&node).is_ok())
            })
        })
        .collect();

    let mut stored = Vec::new();
    for h in handles {
        let (id, ok) = h.join().unwrap();
        if ok {
            stored.push(id);
        }
    }

    println!("VERIFY: {} nodes stored", stored.len());
    assert_eq!(stored.len(), 50, "all 50 nodes should be stored");

    // Verify all nodes are readable
    for id in &stored {
        assert!(
            db.get_node(id).is_ok(),
            "node {} should be readable after concurrent write",
            id
        );
    }
    println!("RESULT: PASSED");
}
