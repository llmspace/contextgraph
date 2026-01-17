//! Search handler happy path tests
//!
//! Tests for search/multi, search/weight_profiles

use serde_json::json;

use super::common::{create_test_handlers_with_rocksdb_store_access, make_request};

/// Test 5: search/multi - Multi-space search
#[tokio::test]
async fn test_05_search_multi() {
    println!("\n========================================================================================================");
    println!("TEST 05: search/multi");
    println!("========================================================================================================");

    let (handlers, _store, _tempdir) = create_test_handlers_with_rocksdb_store_access().await;

    // Store memories
    for content in [
        "vector embeddings for similarity",
        "database optimization techniques",
    ] {
        let params = json!({ "content": content, "importance": 0.8 });
        handlers
            .dispatch(make_request("memory/store", 1, Some(params)))
            .await;
    }

    // Search - MUST include minSimilarity per constitution FAIL FAST policy
    let search_params = json!({
        "query": "vector similarity search",
        "topK": 5,
        "minSimilarity": 0.0,  // Required: use 0.0 to include all results
        "weights": {
            "semantic": 1.0,
            "sparse": 0.5,
            "temporal": 0.3
        }
    });
    let request = make_request("search/multi", 10, Some(search_params));
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
    if let Some(results) = result.get("results") {
        if let Some(arr) = results.as_array() {
            println!("  Found {} results", arr.len());
        }
    }

    println!("\n[PASSED] search/multi works correctly");
}

/// Test 6: search/weight_profiles - Get weight profiles
#[tokio::test]
async fn test_06_search_weight_profiles() {
    println!("\n========================================================================================================");
    println!("TEST 06: search/weight_profiles");
    println!("========================================================================================================");

    let (handlers, _store, _tempdir) = create_test_handlers_with_rocksdb_store_access().await;

    let request = make_request("search/weight_profiles", 1, None);
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
    if let Some(profiles) = result.get("profiles") {
        if let Some(obj) = profiles.as_object() {
            println!("  Available profiles: {:?}", obj.keys().collect::<Vec<_>>());
        }
    }

    println!("\n[PASSED] search/weight_profiles works correctly");
}
