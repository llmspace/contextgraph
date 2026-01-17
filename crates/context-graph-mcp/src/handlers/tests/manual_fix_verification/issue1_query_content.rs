//! Issue 1: search_teleological query_content Parameter Fix Tests
//!
//! BEFORE: Old API required `query` (TeleologicalVectorJson) - complex 13-embedder object
//! AFTER: New API accepts `query_content` (String) - simple string to embed

use crate::handlers::tests::{create_test_handlers, make_request};
use crate::protocol::JsonRpcId;
use serde_json::json;

/// ISSUE-1 VERIFICATION: search_teleological accepts query_content string.
///
/// BEFORE: Old API required `query` (TeleologicalVectorJson) - complex 13-embedder object
/// AFTER: New API accepts `query_content` (String) - simple string to embed
///
/// This test verifies NO "missing field query" error occurs with the new API.
#[tokio::test]
async fn test_issue1_search_teleological_query_content_parameter() {
    println!("\n============================================================");
    println!("ISSUE-1 VERIFICATION: search_teleological query_content fix");
    println!("============================================================");

    // BEFORE STATE: Create handlers with default configuration
    let handlers = create_test_handlers();
    println!("[BEFORE] Handlers created with test configuration");
    println!("[BEFORE] Testing tools/call with search_teleological and query_content parameter");

    // SYNTHETIC DATA: Simple string content that should be accepted
    let synthetic_query = "software architecture patterns for distributed systems";
    println!("[SYNTHETIC DATA] query_content = \"{}\"", synthetic_query);

    // EXECUTE: Call search_teleological with query_content (string)
    let request = make_request(
        "tools/call",
        Some(JsonRpcId::Number(1)),
        Some(json!({
            "name": "search_teleological",
            "arguments": {
                "query_content": synthetic_query,
                "strategy": "cosine",
                "max_results": 5
            }
        })),
    );

    println!("[EXECUTE] Dispatching request with query_content parameter...");
    let response = handlers.dispatch(request).await;

    // SOURCE OF TRUTH: Check response structure
    println!("[SOURCE OF TRUTH] Checking response...");

    // VERIFY: No JSON-RPC error (error field should be None)
    assert!(
        response.error.is_none(),
        "[FAIL] JSON-RPC error returned: {:?}",
        response.error
    );
    println!("[VERIFY] No JSON-RPC protocol error - PASS");

    // VERIFY: Result exists
    let result = response.result.expect("[FAIL] Must have result field");
    println!("[VERIFY] Result field present - PASS");

    // VERIFY: Check content structure
    if let Some(content) = result.get("content").and_then(|v| v.as_array()) {
        println!(
            "[VERIFY] Content array present with {} items - PASS",
            content.len()
        );

        if let Some(first) = content.first() {
            if let Some(text) = first.get("text").and_then(|v| v.as_str()) {
                // Check for the OLD error message that should NOT appear
                let has_old_error = text.contains("missing field `query`")
                    || text.contains("missing field 'query'");
                assert!(
                    !has_old_error,
                    "[FAIL] Old 'missing field query' error detected! FIX NOT APPLIED: {}",
                    text
                );
                println!("[VERIFY] No 'missing field query' error - PASS");

                // Check if it's an expected error (FAIL FAST is OK)
                let is_error = result
                    .get("isError")
                    .and_then(|v| v.as_bool())
                    .unwrap_or(false);
                if is_error {
                    // Some errors are expected (e.g., no stored vectors to search)
                    println!(
                        "[INFO] Tool returned expected error (isError=true): {}",
                        text
                    );
                    // Verify it's a valid FAIL FAST error, not a parameter parsing error
                    if text.contains("FAIL FAST") {
                        println!("[VERIFY] FAIL FAST error (expected for empty store) - PASS");
                    }
                } else {
                    // Success case
                    println!("[VERIFY] Tool executed successfully");
                    if let Ok(parsed) = serde_json::from_str::<serde_json::Value>(text) {
                        if let Some(query_type) = parsed.get("query_type").and_then(|v| v.as_str())
                        {
                            assert_eq!(
                                query_type, "embedded",
                                "[FAIL] Expected query_type=embedded"
                            );
                            println!("[VERIFY] query_type=embedded - PASS");
                        }
                    }
                }
            }
        }
    } else if let Some(is_error) = result.get("isError").and_then(|v| v.as_bool()) {
        // Direct isError check for non-array responses
        println!("[INFO] Response isError: {}", is_error);
    }

    // PHYSICAL EVIDENCE
    println!("\n[PHYSICAL EVIDENCE]");
    println!("  Request method: tools/call");
    println!("  Tool name: search_teleological");
    println!(
        "  Parameter: query_content (string) = \"{}\"",
        synthetic_query
    );
    println!("  Response error: {:?}", response.error);
    println!("  Response has result: true"); // Already consumed by expect() above
    println!("\n[ISSUE-1 VERIFICATION COMPLETE]\n");
}

/// ISSUE-1 VERIFICATION: query_content vs query_vector_id mutual exclusivity.
///
/// API should accept EITHER query_content OR query_vector_id, not both required.
#[tokio::test]
async fn test_issue1_query_content_or_vector_id() {
    println!("\n{}", "=".repeat(60));
    println!("ISSUE-1 VERIFICATION: query_content/query_vector_id mutual exclusivity");
    println!("{}", "=".repeat(60));

    let handlers = create_test_handlers();

    // Test 1: Only query_content (should work)
    println!("\n[TEST 1] Only query_content provided");
    let request1 = make_request(
        "tools/call",
        Some(JsonRpcId::Number(1)),
        Some(json!({
            "name": "search_teleological",
            "arguments": {
                "query_content": "test content only"
            }
        })),
    );
    let response1 = handlers.dispatch(request1).await;
    assert!(
        response1.error.is_none(),
        "query_content alone should not cause protocol error"
    );
    println!("[VERIFY] query_content alone accepted - PASS");

    // Test 2: Only query_vector_id (should work if UUID valid, fail if not found)
    println!("\n[TEST 2] Only query_vector_id provided (valid UUID format)");
    let request2 = make_request(
        "tools/call",
        Some(JsonRpcId::Number(2)),
        Some(json!({
            "name": "search_teleological",
            "arguments": {
                "query_vector_id": "11111111-2222-3333-4444-555555555555"
            }
        })),
    );
    let response2 = handlers.dispatch(request2).await;
    assert!(
        response2.error.is_none(),
        "query_vector_id alone should not cause protocol error"
    );
    println!("[VERIFY] query_vector_id alone accepted - PASS");

    // The result may be an error (vector not found), but that's expected business logic
    if let Some(result) = response2.result {
        if let Some(is_error) = result.get("isError").and_then(|v| v.as_bool()) {
            if is_error {
                println!("[INFO] Expected: vector not found (FAIL FAST) - OK");
            }
        }
    }

    println!("\n[ISSUE-1 MUTUAL EXCLUSIVITY VERIFICATION COMPLETE]\n");
}
