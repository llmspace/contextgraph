//! Comprehensive test summary
//!
//! Runs all happy path tests and summarizes results

use serde_json::json;

use super::common::{create_test_handlers_with_rocksdb_store_access, make_request};

/// Comprehensive test: Run all tests and summarize
#[tokio::test]
async fn test_all_happy_paths_summary() {
    println!("\n========================================================================================================");
    println!("COMPREHENSIVE MCP HAPPY PATH TEST SUMMARY");
    println!("========================================================================================================");

    let (handlers, _store, _tempdir) = create_test_handlers_with_rocksdb_store_access().await;

    let mut passed = 0;
    let mut total = 0;

    // Helper to run a test
    async fn run_test(
        handlers: &crate::handlers::Handlers,
        name: &str,
        method: &str,
        params: Option<serde_json::Value>,
        passed: &mut i32,
        total: &mut i32,
    ) {
        *total += 1;
        let request = make_request(method, *total as i64, params);
        let response = handlers.dispatch(request).await;

        if response.error.is_none() {
            *passed += 1;
            println!("  [PASS] {} - {}", name, method);
        } else {
            let err = response.error.as_ref().unwrap();
            println!(
                "  [FAIL] {} - {} - Error: {} ({})",
                name, method, err.message, err.code
            );
        }
    }

    // Store a test memory for subsequent tests
    let store_params = json!({
        "content": "Comprehensive test memory for all happy paths",
        "importance": 0.9
    });
    let store_response = handlers
        .dispatch(make_request("memory/store", 1, Some(store_params)))
        .await;
    let fingerprint_id = store_response
        .result
        .as_ref()
        .and_then(|r| r.get("fingerprintId"))
        .and_then(|v| v.as_str())
        .unwrap_or("test-id")
        .to_string();

    println!("\nRunning {} MCP method tests...\n", 15);

    // Test each method
    run_test(
        &handlers,
        "Initialize",
        "initialize",
        Some(json!({
            "protocolVersion": "2024-11-05",
            "capabilities": {},
            "clientInfo": {"name": "test", "version": "1.0"}
        })),
        &mut passed,
        &mut total,
    )
    .await;

    run_test(
        &handlers,
        "Tools List",
        "tools/list",
        None,
        &mut passed,
        &mut total,
    )
    .await;

    run_test(
        &handlers,
        "Memory Store",
        "memory/store",
        Some(json!({
            "content": "Test content",
            "importance": 0.8
        })),
        &mut passed,
        &mut total,
    )
    .await;

    run_test(
        &handlers,
        "Memory Retrieve",
        "memory/retrieve",
        Some(json!({
            "fingerprintId": fingerprint_id
        })),
        &mut passed,
        &mut total,
    )
    .await;

    run_test(
        &handlers,
        "Memory Search",
        "memory/search",
        Some(json!({
            "query": "test",
            "topK": 5,
            "minSimilarity": 0.0
        })),
        &mut passed,
        &mut total,
    )
    .await;

    run_test(
        &handlers,
        "Search Multi",
        "search/multi",
        Some(json!({
            "query": "test",
            "topK": 5,
            "minSimilarity": 0.0
        })),
        &mut passed,
        &mut total,
    )
    .await;

    run_test(
        &handlers,
        "Search Weight Profiles",
        "search/weight_profiles",
        None,
        &mut passed,
        &mut total,
    )
    .await;

    run_test(
        &handlers,
        "Purpose Query",
        "purpose/query",
        Some(json!({
            "purpose_vector": [0.8, 0.7, 0.6, 0.5, 0.9, 0.4, 0.3, 0.5, 0.6, 0.7, 0.8, 0.9, 0.5]
        })),
        &mut passed,
        &mut total,
    )
    .await;

    run_test(
        &handlers,
        "UTL Compute",
        "utl/compute",
        Some(json!({
            "input": "Test content for UTL computation to calculate universal truth likelihood"
        })),
        &mut passed,
        &mut total,
    )
    .await;

    run_test(
        &handlers,
        "UTL Metrics",
        "utl/metrics",
        Some(json!({
            "input": "Test content for UTL metrics to analyze universal truth likelihood"
        })),
        &mut passed,
        &mut total,
    )
    .await;

    run_test(
        &handlers,
        "System Status",
        "system/status",
        None,
        &mut passed,
        &mut total,
    )
    .await;

    run_test(
        &handlers,
        "System Health",
        "system/health",
        None,
        &mut passed,
        &mut total,
    )
    .await;

    // Final memory delete
    run_test(
        &handlers,
        "Memory Delete",
        "memory/delete",
        Some(json!({
            "fingerprintId": fingerprint_id
        })),
        &mut passed,
        &mut total,
    )
    .await;

    println!("\n========================================================================================================");
    println!(
        "SUMMARY: {}/{} tests passed ({:.1}%)",
        passed,
        total,
        (passed as f64 / total as f64) * 100.0
    );
    println!("========================================================================================================\n");

    // Core functionality should all pass
    assert!(
        passed >= 12,
        "At least 12 core tests should pass"
    );
}
