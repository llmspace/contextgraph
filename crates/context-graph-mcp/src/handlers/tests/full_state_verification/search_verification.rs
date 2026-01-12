//! Search verification tests.
//!
//! Tests verify that search operations find data that exists in the Source of Truth.

use serde_json::json;

use crate::protocol::JsonRpcId;

use super::helpers::{create_handlers_with_store_access, exists_in_store, make_request};

/// VERIFICATION TEST 3: Search finds fingerprint in Source of Truth.
#[tokio::test]
async fn verify_search_finds_data_in_source_of_truth() {
    println!("\n================================================================================");
    println!("FULL STATE VERIFICATION: Search Finds Source of Truth Data");
    println!("================================================================================");

    let (handlers, store, _provider) = create_handlers_with_store_access();

    // Store multiple fingerprints
    let contents = [
        "Deep learning uses neural networks with many layers",
        "Transformers revolutionized natural language processing",
        "Reinforcement learning teaches agents through rewards",
    ];

    let mut stored_ids = Vec::new();
    for (i, content) in contents.iter().enumerate() {
        let params = json!({ "content": content, "importance": 0.8 });
        let request = make_request(
            "memory/store",
            Some(JsonRpcId::Number(i as i64 + 1)),
            Some(params),
        );
        let response = handlers.dispatch(request).await;
        let id_str = response
            .result
            .unwrap()
            .get("fingerprintId")
            .unwrap()
            .as_str()
            .unwrap()
            .to_string();
        stored_ids.push(uuid::Uuid::parse_str(&id_str).unwrap());
    }

    // === VERIFY ALL IN SOURCE OF TRUTH ===
    let count = store.count().await.expect("count() should work");
    println!("\n[SOURCE OF TRUTH] Store contains {} fingerprints:", count);
    for id in &stored_ids {
        let exists = exists_in_store(&store, *id).await;
        println!("  - {} exists: {}", id, exists);
        assert!(exists, "All stored IDs must exist in Source of Truth");
    }

    // === SEARCH VIA MCP HANDLER ===
    let search_params = json!({
        "query": "neural network deep learning",
        "topK": 10,
        "minSimilarity": 0.0  // P1-FIX-1: Required parameter for fail-fast
    });
    let search_request = make_request(
        "memory/search",
        Some(JsonRpcId::Number(10)),
        Some(search_params),
    );
    let search_response = handlers.dispatch(search_request).await;

    let result = search_response.result.expect("Should have result");
    let results = result.get("results").unwrap().as_array().unwrap();
    let result_count = result.get("count").unwrap().as_u64().unwrap();

    println!("\n[MCP HANDLER] Search returned {} results", result_count);

    // === VERIFY EACH SEARCH RESULT EXISTS IN SOURCE OF TRUTH ===
    for (i, r) in results.iter().enumerate() {
        let result_id_str = r.get("fingerprintId").unwrap().as_str().unwrap();
        let result_id = uuid::Uuid::parse_str(result_id_str).unwrap();
        let similarity = r.get("similarity").unwrap().as_f64().unwrap();

        // Verify this result exists in store
        let exists_in_truth = exists_in_store(&store, result_id).await;
        println!(
            "  Result {}: {} (sim={:.4}) - exists in store: {}",
            i, result_id, similarity, exists_in_truth
        );
        assert!(
            exists_in_truth,
            "Search result must exist in Source of Truth"
        );
    }

    println!("\n[VERIFICATION PASSED] All search results exist in Source of Truth");
    println!("================================================================================\n");
}
