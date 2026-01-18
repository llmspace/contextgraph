//! Basic Storage Example
//!
//! Demonstrates MemoryNode creation, storage in RocksDbMemex, and retrieval.
//! This example shows the fundamental CRUD operations for memory nodes.
//!
//! Run with: `cargo run --package context-graph-storage --example basic_storage`

use context_graph_core::types::MemoryNode;
use context_graph_storage::{Memex, RocksDbMemex};
use tempfile::TempDir;

/// Creates a valid 1536-dimensional normalized embedding vector.
/// This is the required dimension for MemoryNode embeddings.
fn create_valid_embedding() -> Vec<f32> {
    const DIM: usize = 1536;
    let val = 1.0_f32 / (DIM as f32).sqrt();
    vec![val; DIM]
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("=== Basic Storage Example ===\n");

    // Create temp directory for database
    // Using tempfile ensures clean teardown after example runs
    let temp_dir = TempDir::new()?;
    println!("Created temp database at: {:?}", temp_dir.path());

    // Open RocksDbMemex with default configuration
    let memex = RocksDbMemex::open(temp_dir.path())?;
    println!("Opened RocksDbMemex successfully\n");

    // ========================================
    // Example 1: Create and Store a MemoryNode
    // ========================================
    println!("--- Example 1: Store a MemoryNode ---");

    // Create a normalized 1536D embedding
    let embedding = create_valid_embedding();

    // Create MemoryNode with content and embedding
    let mut node = MemoryNode::new(
        "Rust is a systems programming language that runs blazingly fast".to_string(),
        embedding.clone(),
    );

    // Configure node properties
    node.importance = 0.8;
    node.metadata.tags.push("programming".to_string());
    node.metadata.tags.push("rust".to_string());
    node.metadata.tags.push("systems".to_string());
    node.metadata.source = Some("example".to_string());

    // Validate before storing (enforces constitution constraints)
    node.validate()?;
    println!("Node validated successfully");

    // Store the node
    memex.store_node(&node)?;
    println!("Stored node with ID: {}", node.id);
    println!("  Content: {}", node.content);
    println!("  Importance: {}", node.importance);
    println!("  Tags: {:?}\n", node.metadata.tags);

    // ========================================
    // Example 2: Retrieve by ID (Source of Truth)
    // ========================================
    println!("--- Example 2: Retrieve by ID ---");

    let retrieved = memex.get_node(&node.id)?;
    println!("Retrieved node from database:");
    println!("  ID: {}", retrieved.id);
    println!("  Content: {}", retrieved.content);
    println!("  Tags: {:?}", retrieved.metadata.tags);

    // Verify retrieval matches stored data
    assert_eq!(retrieved.id, node.id);
    assert_eq!(retrieved.content, node.content);
    assert_eq!(retrieved.importance, node.importance);
    println!("  Retrieved data matches stored data\n");

    // ========================================
    // Example 3: Query by Tag (Secondary Index)
    // ========================================
    println!("--- Example 3: Query by Tag ---");

    let rust_node_ids = memex.query_by_tag("rust", Some(10))?;
    println!("Found {} node(s) with 'rust' tag", rust_node_ids.len());

    // Fetch full nodes to display content and tags
    for node_id in &rust_node_ids {
        let n = memex.get_node(node_id)?;
        println!(
            "  - {} (tags: {:?})",
            n.content.chars().take(30).collect::<String>(),
            n.metadata.tags
        );
    }
    assert!(
        !rust_node_ids.is_empty(),
        "Should find at least one rust-tagged node"
    );
    println!("  Tag queries working correctly\n");

    // ========================================
    // Example 4: Update a Node
    // ========================================
    println!("--- Example 4: Update Node ---");

    let mut updated_node = retrieved.clone();
    updated_node.importance = 0.95;
    updated_node.metadata.tags.push("updated".to_string());
    updated_node.record_access(); // Track access for decay calculation

    memex.update_node(&updated_node)?;
    println!("Updated node importance to {}", updated_node.importance);

    // Verify update in source of truth
    let re_retrieved = memex.get_node(&node.id)?;
    assert_eq!(re_retrieved.importance, 0.95);
    assert!(re_retrieved.metadata.tags.contains(&"updated".to_string()));
    assert!(re_retrieved.access_count >= 1);
    println!("  Update verified in database\n");

    // ========================================
    // Example 5: Delete a Node (Soft Delete)
    // ========================================
    println!("--- Example 5: Delete Node (Soft Delete) ---");

    // Create another node to delete
    let mut to_delete = MemoryNode::new(
        "This node will be deleted".to_string(),
        create_valid_embedding(),
    );
    to_delete.validate()?;
    memex.store_node(&to_delete)?;

    // Soft delete the node (SEC-06: 30-day recovery)
    memex.delete_node(&to_delete.id, true)?;
    println!("Soft deleted node: {}", to_delete.id);

    // Soft delete still allows retrieval but marks as deleted
    let soft_deleted = memex.get_node(&to_delete.id)?;
    println!("  Soft deleted node still retrievable");
    println!("  metadata.deleted = {}", soft_deleted.metadata.deleted);
    assert!(
        soft_deleted.metadata.deleted,
        "Node should be marked as deleted"
    );
    println!("  Soft delete verified\n");

    // ========================================
    // Example 6: Embedding Storage
    // ========================================
    println!("--- Example 6: Embedding Operations ---");

    // Store an embedding separately (useful for vector search)
    let node_id = node.id;
    memex.store_embedding(&node_id, &embedding)?;
    println!("Stored embedding for node {}", node_id);

    // Retrieve embedding
    let retrieved_embedding = memex.get_embedding(&node_id)?;
    println!(
        "Retrieved embedding with {} dimensions",
        retrieved_embedding.len()
    );
    assert_eq!(retrieved_embedding.len(), 1536);

    // Check if embedding exists
    let exists = memex.embedding_exists(&node_id)?;
    assert!(exists);
    println!("  Embedding exists check passed\n");

    // ========================================
    // Example 7: Database Health Check
    // ========================================
    println!("--- Example 7: Health Check ---");

    // Use the Memex trait for health_check to get StorageHealth
    let memex_trait: &dyn Memex = &memex;
    let health = memex_trait.health_check()?;
    println!("Database Health Status:");
    println!("  Is Healthy: {}", health.is_healthy);
    println!("  Node Count: {}", health.node_count);
    println!("  Edge Count: {}", health.edge_count);
    println!("  Storage Bytes: {}", health.storage_bytes);

    assert!(health.is_healthy, "Database should be healthy");
    println!("  Health check verified\n");

    // ========================================
    // Example 8: Flush to Disk
    // ========================================
    println!("--- Example 8: Flush to Disk ---");

    memex.flush_all()?;
    println!("Flushed all data to disk");
    println!("  Flush completed successfully\n");

    println!("=== All Examples Completed Successfully ===");
    println!("Database will be cleaned up when temp directory is dropped");

    Ok(())
}
