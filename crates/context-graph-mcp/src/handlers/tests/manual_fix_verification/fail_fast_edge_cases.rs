//! FAIL FAST Edge Cases Verification
//!
//! Tests for immediate failure on invalid inputs following the FAIL FAST pattern.

use crate::handlers::tests::{create_test_handlers, make_request};
use crate::protocol::JsonRpcId;
use serde_json::json;

/// FAIL FAST: Empty query_content should fail immediately.
#[tokio::test]
async fn test_fail_fast_empty_query_content() {
    println!("\n{}", "=".repeat(60));
    println!("FAIL FAST VERIFICATION: Empty query_content");
    println!("{}", "=".repeat(60));

    let handlers = create_test_handlers();

    let request = make_request(
        "tools/call",
        Some(JsonRpcId::Number(1)),
        Some(json!({
            "name": "search_teleological",
            "arguments": {
                "query_content": ""
            }
        })),
    );

    println!("[EXECUTE] Calling search_teleological with empty query_content...");
    let response = handlers.dispatch(request).await;

    // Should not have protocol error
    assert!(response.error.is_none(), "Should not have protocol error");
    let result = response.result.expect("Must have result");

    // VERIFY: Should be isError=true with FAIL FAST message
    let is_error = result
        .get("isError")
        .and_then(|v| v.as_bool())
        .unwrap_or(false);
    assert!(is_error, "[FAIL] Empty query_content should trigger error");
    println!("[VERIFY] isError=true for empty content - PASS");

    if let Some(content) = result.get("content").and_then(|v| v.as_array()) {
        if let Some(first) = content.first() {
            if let Some(text) = first.get("text").and_then(|v| v.as_str()) {
                let has_fail_fast = text.contains("FAIL FAST") || text.contains("empty");
                assert!(has_fail_fast, "[FAIL] Should mention FAIL FAST or empty");
                println!("[VERIFY] FAIL FAST message for empty content - PASS");
                println!("[EVIDENCE] Error text: {}", text);
            }
        }
    }

    println!("\n[FAIL FAST empty query_content VERIFICATION COMPLETE]\n");
}

/// FAIL FAST: Invalid UUID for query_vector_id should fail immediately.
#[tokio::test]
async fn test_fail_fast_invalid_uuid() {
    println!("\n{}", "=".repeat(60));
    println!("FAIL FAST VERIFICATION: Invalid UUID");
    println!("{}", "=".repeat(60));

    let handlers = create_test_handlers();

    let request = make_request(
        "tools/call",
        Some(JsonRpcId::Number(1)),
        Some(json!({
            "name": "search_teleological",
            "arguments": {
                "query_vector_id": "not-a-valid-uuid-format"
            }
        })),
    );

    println!("[EXECUTE] Calling search_teleological with invalid UUID...");
    let response = handlers.dispatch(request).await;

    assert!(response.error.is_none(), "Should not have protocol error");
    let result = response.result.expect("Must have result");

    // VERIFY: Should be isError=true with FAIL FAST message
    let is_error = result
        .get("isError")
        .and_then(|v| v.as_bool())
        .unwrap_or(false);
    assert!(is_error, "[FAIL] Invalid UUID should trigger error");
    println!("[VERIFY] isError=true for invalid UUID - PASS");

    if let Some(content) = result.get("content").and_then(|v| v.as_array()) {
        if let Some(first) = content.first() {
            if let Some(text) = first.get("text").and_then(|v| v.as_str()) {
                let has_fail_fast = text.contains("FAIL FAST");
                assert!(has_fail_fast, "[FAIL] Should mention FAIL FAST");
                println!("[VERIFY] FAIL FAST message for invalid UUID - PASS");
                println!("[EVIDENCE] Error text: {}", text);
            }
        }
    }

    println!("\n[FAIL FAST invalid UUID VERIFICATION COMPLETE]\n");
}

/// FAIL FAST: Neither query_content nor query_vector_id provided.
#[tokio::test]
async fn test_fail_fast_no_query_provided() {
    println!("\n{}", "=".repeat(60));
    println!("FAIL FAST VERIFICATION: No query parameter provided");
    println!("{}", "=".repeat(60));

    let handlers = create_test_handlers();

    let request = make_request(
        "tools/call",
        Some(JsonRpcId::Number(1)),
        Some(json!({
            "name": "search_teleological",
            "arguments": {
                "strategy": "cosine",
                "max_results": 10
            }
        })),
    );

    println!("[EXECUTE] Calling search_teleological without query parameter...");
    let response = handlers.dispatch(request).await;

    assert!(response.error.is_none(), "Should not have protocol error");
    let result = response.result.expect("Must have result");

    // VERIFY: Should be isError=true with FAIL FAST message
    let is_error = result
        .get("isError")
        .and_then(|v| v.as_bool())
        .unwrap_or(false);
    assert!(is_error, "[FAIL] Missing query should trigger error");
    println!("[VERIFY] isError=true for missing query - PASS");

    if let Some(content) = result.get("content").and_then(|v| v.as_array()) {
        if let Some(first) = content.first() {
            if let Some(text) = first.get("text").and_then(|v| v.as_str()) {
                let has_fail_fast = text.contains("FAIL FAST");
                let mentions_params =
                    text.contains("query_content") || text.contains("query_vector_id");
                assert!(has_fail_fast, "[FAIL] Should mention FAIL FAST");
                assert!(mentions_params, "[FAIL] Should mention required parameters");
                println!("[VERIFY] FAIL FAST message mentions required params - PASS");
                println!("[EVIDENCE] Error text: {}", text);
            }
        }
    }

    println!("\n[FAIL FAST no query VERIFICATION COMPLETE]\n");
}
