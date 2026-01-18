//! Memex trait compliance tests.
//!
//! Tests that RocksDbMemex implements the Memex trait correctly.

use super::common::{create_node_with_content, create_test_node, setup_db};
use context_graph_storage::Memex;

#[test]
fn test_memex_trait() {
    let (db, _tmp) = setup_db();
    let memex: &dyn Memex = &db;

    println!("=== MEMEX TRAIT TEST ===");
    let node = create_test_node();
    let node_id = node.id;

    // Store via trait
    memex.store_node(&node).expect("store via trait");

    // Get via trait
    let retrieved = memex.get_node(&node_id).expect("get via trait");
    println!("VERIFY: roundtrip via trait");
    assert_eq!(retrieved.id, node.id);
    assert_eq!(retrieved.content, node.content);

    // Health check via trait
    let health = memex.health_check().expect("health check");
    println!(
        "VERIFY: is_healthy={}, node_count={}",
        health.is_healthy, health.node_count
    );
    assert!(health.is_healthy, "health check should pass");
    assert!(health.node_count >= 1, "should have at least 1 node");

    println!("RESULT: PASSED");
}

#[test]
fn test_memex_trait_object_safety() {
    let (db, _tmp) = setup_db();

    println!("=== MEMEX TRAIT OBJECT SAFETY TEST ===");
    println!("BEFORE: Creating Box<dyn Memex>");

    // This line compiles only if trait is object-safe
    let boxed: Box<dyn Memex> = Box::new(db);

    // Use the boxed trait object
    let node = create_test_node();
    boxed.store_node(&node).expect("store via Box<dyn Memex>");

    let retrieved = boxed.get_node(&node.id).expect("get via Box<dyn Memex>");
    assert_eq!(retrieved.id, node.id);

    println!("AFTER: Box<dyn Memex> operations successful");
    println!("RESULT: PASSED");
}

#[test]
fn test_health_check() {
    let (db, _tmp) = setup_db();

    println!("=== HEALTH CHECK TEST ===");

    // Cast to Memex trait to get StorageHealth (inherent method returns ())
    let memex: &dyn Memex = &db;

    // Empty database health
    let health = memex.health_check().expect("health check");
    println!(
        "BEFORE: is_healthy={}, node_count={}, edge_count={}, storage_bytes={}",
        health.is_healthy, health.node_count, health.edge_count, health.storage_bytes
    );
    assert!(health.is_healthy);

    // Add some data
    for i in 0..10 {
        let node = create_node_with_content(&format!("Health {}", i));
        db.store_node(&node).expect("store");
    }

    let health_after = memex.health_check().expect("health check after");
    println!(
        "AFTER: is_healthy={}, node_count={}, edge_count={}, storage_bytes={}",
        health_after.is_healthy,
        health_after.node_count,
        health_after.edge_count,
        health_after.storage_bytes
    );
    assert!(health_after.is_healthy);
    assert!(
        health_after.node_count >= 10,
        "should count at least 10 nodes"
    );

    println!("RESULT: PASSED");
}
