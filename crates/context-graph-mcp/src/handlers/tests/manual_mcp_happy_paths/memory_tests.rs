//! Memory handler happy path tests
//!
//! Tests for memory/store, memory/retrieve, memory/search, memory/delete

use serde_json::json;

use super::common::{create_test_handlers_with_rocksdb_store_access, make_request};

/// Test 1: memory/store - Store a new memory
#[tokio::test]
async fn test_01_memory_store() {
    println!("\n========================================================================================================");
    println!("TEST 01: memory/store");
    println!("========================================================================================================");

    let (handlers, _store, _tempdir) = create_test_handlers_with_rocksdb_store_access().await;

    let params = json!({
        "content": "Test memory for happy path validation",
        "importance": 0.85
    });

    let request = make_request("memory/store", 1, Some(params));
    let response = handlers.dispatch(request).await;

    println!(
        "Response: {}",
        serde_json::to_string_pretty(&response).unwrap()
    );

    assert!(
        response.error.is_none(),
        "Should not have error: {:?}",
        response.error
    );
    let result = response.result.expect("Should have result");

    // Verify required fields from memory/store response
    // API returns: fingerprintId, embeddingLatencyMs, storageLatencyMs, embedderCount
    let fingerprint_id = result
        .get("fingerprintId")
        .expect("Should have fingerprintId");
    let embedder_count = result
        .get("embedderCount")
        .expect("Should have embedderCount");

    println!("\n[VERIFICATION]");
    println!("  fingerprintId: {}", fingerprint_id);
    println!("  embedderCount: {}", embedder_count);
    println!(
        "  embeddingLatencyMs: {}",
        result.get("embeddingLatencyMs").unwrap_or(&json!(0))
    );
    println!(
        "  storageLatencyMs: {}",
        result.get("storageLatencyMs").unwrap_or(&json!(0))
    );

    assert!(fingerprint_id.is_string(), "fingerprintId should be string");
    assert_eq!(
        embedder_count.as_u64().unwrap(),
        13,
        "Should have 13 embedders"
    );
    println!("\n[PASSED] memory/store works correctly");
}

/// Test 2: memory/retrieve - Retrieve an existing memory
#[tokio::test]
async fn test_02_memory_retrieve() {
    println!("\n========================================================================================================");
    println!("TEST 02: memory/retrieve");
    println!("========================================================================================================");

    let (handlers, _store, _tempdir) = create_test_handlers_with_rocksdb_store_access().await;

    // First store a memory to retrieve
    let store_params = json!({
        "content": "Memory to retrieve for testing",
        "importance": 0.9
    });
    let store_response = handlers
        .dispatch(make_request("memory/store", 1, Some(store_params)))
        .await;
    let fingerprint_id = store_response
        .result
        .unwrap()
        .get("fingerprintId")
        .unwrap()
        .as_str()
        .unwrap()
        .to_string();

    println!("Stored fingerprint: {}", fingerprint_id);

    // Now retrieve it
    let retrieve_params = json!({
        "fingerprintId": fingerprint_id
    });
    let request = make_request("memory/retrieve", 2, Some(retrieve_params));
    let response = handlers.dispatch(request).await;

    println!(
        "Response: {}",
        serde_json::to_string_pretty(&response).unwrap()
    );

    assert!(
        response.error.is_none(),
        "Should not have error: {:?}",
        response.error
    );
    let result = response.result.expect("Should have result");
    let fingerprint = result.get("fingerprint").expect("Should have fingerprint");

    println!("\n[VERIFICATION]");
    println!("  id: {}", fingerprint.get("id").unwrap());
    println!(
        "  alignmentScore: {}",
        fingerprint.get("alignmentScore").unwrap()
    );
    println!("  accessCount: {}", fingerprint.get("accessCount").unwrap());
    println!(
        "  contentHashHex: {}",
        fingerprint.get("contentHashHex").unwrap()
    );

    // Check purpose vector
    if let Some(pv) = fingerprint.get("purposeVector") {
        println!(
            "  purposeVector.coherence: {}",
            pv.get("coherence").unwrap_or(&json!(null))
        );
        println!(
            "  purposeVector.dominantEmbedder: {}",
            pv.get("dominantEmbedder").unwrap_or(&json!(null))
        );
    }

    println!("\n[PASSED] memory/retrieve works correctly");
}

/// Test 3: memory/search - Search for memories
#[tokio::test]
async fn test_03_memory_search() {
    println!("\n========================================================================================================");
    println!("TEST 03: memory/search");
    println!("========================================================================================================");

    let (handlers, _store, _tempdir) = create_test_handlers_with_rocksdb_store_access().await;

    // Store some memories first
    for (i, content) in [
        "neural network optimization",
        "distributed systems consensus",
        "machine learning algorithms",
    ]
    .iter()
    .enumerate()
    {
        let params = json!({ "content": content, "importance": 0.8 });
        handlers
            .dispatch(make_request("memory/store", i as i64, Some(params)))
            .await;
    }

    // Search - MUST include minSimilarity per constitution FAIL FAST policy
    let search_params = json!({
        "query": "neural network",
        "topK": 5,
        "minSimilarity": 0.0  // Required: use 0.0 to include all results
    });
    let request = make_request("memory/search", 10, Some(search_params));
    let response = handlers.dispatch(request).await;

    println!(
        "Response: {}",
        serde_json::to_string_pretty(&response).unwrap()
    );

    assert!(
        response.error.is_none(),
        "Should not have error: {:?}",
        response.error
    );
    let result = response.result.expect("Should have result");
    let results = result.get("results").expect("Should have results");

    println!("\n[VERIFICATION]");
    if let Some(arr) = results.as_array() {
        println!("  Found {} results", arr.len());
        for (i, r) in arr.iter().take(3).enumerate() {
            println!(
                "  [{}] id={}, similarity={}",
                i,
                r.get("fingerprintId").unwrap_or(&json!("?")),
                r.get("similarity").unwrap_or(&json!(0))
            );
        }
    }

    println!("\n[PASSED] memory/search works correctly");
}

/// Test 4: memory/delete - Delete a memory
#[tokio::test]
async fn test_04_memory_delete() {
    println!("\n========================================================================================================");
    println!("TEST 04: memory/delete");
    println!("========================================================================================================");

    let (handlers, store, _tempdir) = create_test_handlers_with_rocksdb_store_access().await;

    // First store a memory
    let store_params = json!({
        "content": "Memory to be deleted",
        "importance": 0.5
    });
    let store_response = handlers
        .dispatch(make_request("memory/store", 1, Some(store_params)))
        .await;
    let fingerprint_id = store_response
        .result
        .unwrap()
        .get("fingerprintId")
        .unwrap()
        .as_str()
        .unwrap()
        .to_string();

    println!("Stored fingerprint: {}", fingerprint_id);

    let count_before = store.count().await.unwrap();
    println!("Count before delete: {}", count_before);

    // Hard delete (soft=false) to actually remove from count
    // Soft delete is default and marks as deleted but doesn't remove from count
    let delete_params = json!({
        "fingerprintId": fingerprint_id,
        "soft": false  // Hard delete to actually remove from storage
    });
    let request = make_request("memory/delete", 2, Some(delete_params));
    let response = handlers.dispatch(request).await;

    println!(
        "Response: {}",
        serde_json::to_string_pretty(&response).unwrap()
    );

    assert!(
        response.error.is_none(),
        "Should not have error: {:?}",
        response.error
    );
    let result = response.result.expect("Should have result");

    let count_after = store.count().await.unwrap();
    println!("Count after delete: {}", count_after);

    println!("\n[VERIFICATION]");
    println!(
        "  deleted: {}",
        result.get("deleted").unwrap_or(&json!(false))
    );
    println!(
        "  deleteType: {}",
        result.get("deleteType").unwrap_or(&json!("?"))
    );
    println!("  count decreased: {} -> {}", count_before, count_after);

    assert!(
        result
            .get("deleted")
            .and_then(|v| v.as_bool())
            .unwrap_or(false),
        "Delete should return true"
    );
    assert_eq!(
        result
            .get("deleteType")
            .and_then(|v| v.as_str())
            .unwrap_or("?"),
        "hard",
        "Should be hard delete"
    );
    assert!(
        count_after < count_before,
        "Count should decrease after hard delete"
    );
    println!("\n[PASSED] memory/delete works correctly");
}
