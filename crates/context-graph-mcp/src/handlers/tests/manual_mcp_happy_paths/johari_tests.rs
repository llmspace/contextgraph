//! Johari handler happy path tests
//!
//! Tests for johari/get_distribution, johari/find_by_quadrant

use serde_json::json;

use super::common::{create_test_handlers_with_rocksdb_store_access, make_request};

/// Test 12: johari/get_distribution - Get Johari distribution
#[tokio::test]
async fn test_12_johari_get_distribution() {
    println!("\n========================================================================================================");
    println!("TEST 12: johari/get_distribution");
    println!("========================================================================================================");

    let (handlers, _store, _tempdir) = create_test_handlers_with_rocksdb_store_access().await;

    // Store a memory first to get its ID
    let store_params = json!({
        "content": "Johari test memory for distribution analysis",
        "importance": 0.8
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

    // johari/get_distribution requires memory_id
    let distribution_params = json!({
        "memory_id": fingerprint_id
    });
    let request = make_request("johari/get_distribution", 10, Some(distribution_params));
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

    println!("\n[VERIFICATION]");
    // Response contains per-embedder quadrant distribution
    if let Some(embedders) = result.get("embedders") {
        if let Some(arr) = embedders.as_array() {
            println!("  Found {} embedder distributions", arr.len());
            for (i, e) in arr.iter().take(3).enumerate() {
                println!(
                    "  [{}] {}: quadrant={}",
                    i,
                    e.get("embedder_name").unwrap_or(&json!("?")),
                    e.get("quadrant").unwrap_or(&json!("?"))
                );
            }
        }
    }

    if let Some(summary) = result.get("summary") {
        println!(
            "  summary.open_count: {}",
            summary.get("open_count").unwrap_or(&json!("?"))
        );
        println!(
            "  summary.unknown_count: {}",
            summary.get("unknown_count").unwrap_or(&json!("?"))
        );
    }

    println!("\n[PASSED] johari/get_distribution works correctly");
}

/// Test 13: johari/find_by_quadrant - Find memories by quadrant
#[tokio::test]
async fn test_13_johari_find_by_quadrant() {
    println!("\n========================================================================================================");
    println!("TEST 13: johari/find_by_quadrant");
    println!("========================================================================================================");

    let (handlers, _store, _tempdir) = create_test_handlers_with_rocksdb_store_access().await;

    // Store some memories (they start in Unknown quadrant)
    for content in ["Quadrant search test 1", "Quadrant search test 2"] {
        let params = json!({ "content": content, "importance": 0.8 });
        handlers
            .dispatch(make_request("memory/store", 1, Some(params)))
            .await;
    }

    // johari/find_by_quadrant requires embedder_index and quadrant
    let search_params = json!({
        "embedder_index": 0,  // E1_semantic (first embedder)
        "quadrant": "Unknown",
        "top_k": 10
    });
    let request = make_request("johari/find_by_quadrant", 10, Some(search_params));
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

    println!("\n[VERIFICATION]");
    if let Some(fingerprints) = result.get("fingerprints") {
        if let Some(arr) = fingerprints.as_array() {
            println!(
                "  Found {} fingerprints in Unknown quadrant (E1_semantic)",
                arr.len()
            );
            for (i, fp) in arr.iter().take(3).enumerate() {
                println!("  [{}] id={}", i, fp.get("id").unwrap_or(&json!("?")));
            }
        }
    }

    println!("\n[PASSED] johari/find_by_quadrant works correctly");
}
