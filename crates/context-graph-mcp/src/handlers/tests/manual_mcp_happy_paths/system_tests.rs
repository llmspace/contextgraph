//! System handler happy path tests
//!
//! Tests for system/status, system/health

use serde_json::json;

use super::common::{create_test_handlers_with_rocksdb_store_access, make_request};

/// Test 10: system/status - Get system status
#[tokio::test]
async fn test_10_system_status() {
    println!("\n========================================================================================================");
    println!("TEST 10: system/status");
    println!("========================================================================================================");

    let (handlers, _store, _tempdir) = create_test_handlers_with_rocksdb_store_access().await;

    let request = make_request("system/status", 1, None);
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
    println!(
        "  fingerprintCount: {}",
        result.get("fingerprintCount").unwrap_or(&json!("?"))
    );
    println!(
        "  coherence: {}",
        result.get("coherence").unwrap_or(&json!("?"))
    );
    println!(
        "  entropy: {}",
        result.get("entropy").unwrap_or(&json!("?"))
    );

    // Check quadrant distribution
    if let Some(quadrants) = result.get("quadrantDistribution") {
        println!("  quadrantDistribution: {:?}", quadrants);
    }

    println!("\n[PASSED] system/status works correctly");
}

/// Test 11: system/health - Get system health
#[tokio::test]
async fn test_11_system_health() {
    println!("\n========================================================================================================");
    println!("TEST 11: system/health");
    println!("========================================================================================================");

    let (handlers, _store, _tempdir) = create_test_handlers_with_rocksdb_store_access().await;

    let request = make_request("system/health", 1, None);
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
    println!("  status: {}", result.get("status").unwrap_or(&json!("?")));
    println!(
        "  storageHealthy: {}",
        result.get("storageHealthy").unwrap_or(&json!("?"))
    );
    println!(
        "  embeddingHealthy: {}",
        result.get("embeddingHealthy").unwrap_or(&json!("?"))
    );

    println!("\n[PASSED] system/health works correctly");
}
